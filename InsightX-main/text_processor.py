# text_processor.py
# Text cleaning and processing functions

import string
import re
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

@st.cache_data
def clean_text(text):
    """Text cleaning function"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    domain_stopwords = {'algorithm', 'method', 'problem', 'model', 'data', 'result', 'approach', 'paper','user'}
    stop_words.update(domain_stopwords)
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def extract_future_scope(abstract):
    """Extract future scope or conclusion section from abstract"""
    if not abstract:
        return None
    
    text_lower = abstract.lower()
    
    future_patterns = [
        r'(future work.*?)(?=\n\n|\n[A-Z]|$)',
        r'(future research.*?)(?=\n\n|\n[A-Z]|$)',
        r'(future directions.*?)(?=\n\n|\n[A-Z]|$)',
        r'(limitations and future work.*?)(?=\n\n|\n[A-Z]|$)',
        r'(conclusion and future work.*?)(?=\n\n|\n[A-Z]|$)',
        r'(we conclude.*?)(?=\n\n|\n[A-Z]|$)',
        r'(in conclusion.*?)(?=\n\n|\n[A-Z]|$)',
        r'(to conclude.*?)(?=\n\n|\n[A-Z]|$)',
    ]
    
    conclusion_patterns = [
        r'(conclusion.*?)(?=\n\n|\n[A-Z]|$)',
        r'(discussion and conclusion.*?)(?=\n\n|\n[A-Z]|$)',
        r'(summary and conclusion.*?)(?=\n\n|\n[A-Z]|$)',
    ]
    
    for pattern in future_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            start_pos = abstract.lower().find(match.group(1))
            if start_pos != -1:
                end_pos = start_pos + len(match.group(1))
                return abstract[start_pos:end_pos].strip()
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            start_pos = abstract.lower().find(match.group(1))
            if start_pos != -1:
                end_pos = start_pos + len(match.group(1))
                return abstract[start_pos:end_pos].strip()
    
    return None