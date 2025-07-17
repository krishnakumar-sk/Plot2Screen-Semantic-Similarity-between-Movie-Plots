import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import spacy
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords 
from fuzzywuzzy import process
import chardet
import os
from collections import defaultdict
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def load_css():
    st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        h1 {color: #ff4b4b;}
        .sidebar .sidebar-content {background-color: #ffffff;}
        .stButton>button {border-radius: 8px; padding: 10px 24px; background-color: #ff4b4b; color: white;}
        .stSelectbox div[data-baseweb="select"] {border-radius: 8px;}
        .movie-card {border-radius: 8px; padding: 15px; margin-bottom: 15px; 
                     background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .similarity-score {color: #ff4b4b; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']
def load_default_dataset():
    file_path = "Indian_movies.csv"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error reading file with {encoding}: {str(e)}")
            continue
    else:
        try:
            detected_encoding = detect_file_encoding(file_path)
            df = pd.read_csv(file_path, encoding=detected_encoding)
        except Exception as e:
            st.error(f"Failed to load dataset with detected encoding: {str(e)}")
            return None
    df.columns = df.columns.str.lower().str.strip()
    title_col, plot_col = None, None
    for col in df.columns:
        if not title_col and ('title' in col or 'name' in col or 'movie' in col):
            title_col = col
        if not plot_col and ('plot' in col or 'summary' in col or 'synopsis' in col or 'description' in col):
            plot_col = col
    if not title_col or not plot_col:
        st.error("Required columns not found. Dataset needs 'title' and 'plot' columns.")
        st.write("Available columns:", df.columns.tolist())
        return None
    df = df.rename(columns={title_col: 'title', plot_col: 'plot'})
    df = df[['title', 'plot']].copy()
    df = df.dropna(subset=['title', 'plot'])
    df = df[df['plot'].astype(str).str.strip() != '']
    if len(df) > 10000:
        df = df.sample(10000, random_state=42).reset_index(drop=True)
    return df
def advanced_preprocess(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:  
                tokens.append(token.lemma_)
    return ' '.join(tokens)
@st.cache_resource
def create_memory_efficient_models(df):
    df['processed_plot'] = df['plot'].apply(advanced_preprocess)
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  
        min_df=5,            
        max_df=0.85,
        max_features=5000   
    )    
    tfidf_matrix = tfidf.fit_transform(df['processed_plot'])
    svd = TruncatedSVD(n_components=100, random_state=42)  
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    sentences = [text.split() for text in df['processed_plot']]
    word2vec_model = Word2Vec(
        sentences, 
        vector_size=50, 
        window=5, 
        min_count=3,     
        workers=4
    )
    def document_vector(doc):
        doc = [word for word in doc.split() if word in word2vec_model.wv]
        if len(doc) == 0:
            return np.zeros(50)
        return np.mean(word2vec_model.wv[doc], axis=0)
    w2v_matrix = np.array([document_vector(text) for text in df['processed_plot']])
    tfidf_nn = NearestNeighbors(n_neighbors=50, metric='cosine').fit(tfidf_reduced)
    w2v_nn = NearestNeighbors(n_neighbors=50, metric='cosine').fit(w2v_matrix)
    return {
        'df': df,
        'tfidf_nn': tfidf_nn,
        'w2v_nn': w2v_nn,
        'tfidf_reduced': tfidf_reduced,
        'w2v_matrix': w2v_matrix,
        'word2vec_model': word2vec_model
    }
def fuzzy_match_title(input_title, titles):
    matches = process.extractOne(input_title, titles)
    if matches and matches[1] > 70:
        return matches[0]
    return None
def get_recommendations(input_title, models, top_n=10, tfidf_weight=0.7, w2v_weight=0.3):
    df = models['df']
    try:
        exact_matches = df[df['title'].str.lower() == input_title.lower()]
        if len(exact_matches) == 0:
            matched_title = fuzzy_match_title(input_title, df['title'].tolist())
            if matched_title:
                exact_matches = df[df['title'].str.lower() == matched_title.lower()]
        if len(exact_matches) == 0:
            return None, "Movie not found in dataset. Please try another title."
        idx = exact_matches.index[0]
        _, tfidf_indices = models['tfidf_nn'].kneighbors([models['tfidf_reduced'][idx]])
        _, w2v_indices = models['w2v_nn'].kneighbors([models['w2v_matrix'][idx]])
        combined_scores = defaultdict(float)
        for i, pos in enumerate(tfidf_indices[0]):
            combined_scores[pos] += tfidf_weight * (1 - i/len(tfidf_indices[0]))  
        for i, pos in enumerate(w2v_indices[0]):
            combined_scores[pos] += w2v_weight * (1 - i/len(w2v_indices[0]))
        combined_scores.pop(idx, None)
        top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = df.iloc[[i[0] for i in top_indices]].copy()
        recommendations['similarity_score'] = [i[1] for i in top_indices]
        return recommendations, exact_matches.iloc[0]['title']
    except Exception as e:
        return None, f"Error getting recommendations: {str(e)}"
def main():
    st.set_page_config(
        page_title="Indian Movie Recommender",
        layout="wide",
        page_icon="🎬"
    )
    load_css()
    st.title("🎬Plot 2 Screen")
    st.markdown("Semantic similarities of movies through Plot analysis")
    with st.spinner("Loading movie database and building models..."):
        df = load_default_dataset()
        if df is None:
            return
        models = create_memory_efficient_models(df)    
    col1, col2 = st.columns([1, 3])    
    with col1:
        st.subheader("Find Similar Movies")        
        movie_list = models['df']['title'].unique()
        input_title = st.selectbox(
            "Select a movie:",
            movie_list,
            index=0,
            help="Start typing to search Indian movies"
        )        
        top_n = st.slider(
            "Number of recommendations:",
            3, 20, 10,
            help="Adjust how many similar movies to show"
        )        
        tfidf_weight = st.slider(
            "Content similarity weight:",
            0.1, 0.9, 0.7,
            help="How much to weight keyword-based similarity"
        )
        w2v_weight = 1.0 - tfidf_weight        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Finding similar movies..."):
                recommendations, matched_title = get_recommendations(
                    input_title,
                    models,
                    top_n,
                    tfidf_weight,
                    w2v_weight
                )                
                if recommendations is not None:
                    st.session_state.recommendations = recommendations
                    st.session_state.matched_title = matched_title
                else:
                    st.error(matched_title)    
    with col2:
        if 'recommendations' in st.session_state:
            st.subheader(f"Movies similar to: {st.session_state.matched_title}")
            st.write(f"Similarity scores range: {st.session_state.recommendations['similarity_score'].min():.2f} to {st.session_state.recommendations['similarity_score'].max():.2f}")
            cols = st.columns(2)
            for i, (_, row) in enumerate(st.session_state.recommendations.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>{i+1}. {row['title']}</h4>
                        <div class="similarity-score">Similarity: {row['similarity_score']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)                    
                    with st.expander("View Plot Summary"):
                        st.write(row['plot'][:500] + "..." if len(row['plot']) > 500 else row['plot'])
            st.markdown("---")
            st.caption("Recommendations generated using memory-efficient hybrid NLP analysis")
if __name__ == "__main__":
    main()