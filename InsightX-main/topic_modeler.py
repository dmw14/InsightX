# topic_modeler.py
# LDA modeling and topic analysis

import streamlit as st
from gensim import corpora
from gensim.models import LdaModel
import numpy as np
from text_processor import clean_text
from groq_client import generate_with_groq
from config import NUM_TOPICS, NUM_KEYWORDS

# topic_modeler.py - Update the perform_topic_modeling function

def perform_topic_modeling(df, current_domain, enable_ai_features, groq_client):
    """Perform LDA topic modeling and AI-powered topic naming"""
    with st.spinner("Preprocessing and modeling with AI enhancements..."):
        publication_years = df['year'].tolist()
        abstracts = df['abstract'].tolist()
        cleaned_abstracts = [clean_text(abstract) for abstract in abstracts]
        
        # LDA Setup
        tokenized_texts = [text.split() for text in cleaned_abstracts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # LDA Model
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, passes=15, random_state=42)
        dominant_topics = [max(lda_model.get_document_topics(doc_bow), key=lambda x: x[1])[0] for doc_bow in corpus]
        df['Dominant_Topic_ID'] = dominant_topics
        
        # FIX: Ensure we always have topic names, even if AI fails
        if enable_ai_features and groq_client:
            with st.spinner("🤖 Generating AI-powered topic names..."):
                try:
                    topic_keywords = extract_topic_keywords(lda_model, dictionary, NUM_TOPICS, NUM_KEYWORDS)
                    ai_topic_names = generate_ai_topic_names(current_domain, range(NUM_TOPICS), topic_keywords)
                    df['Dominant_Topic_Name'] = df['Dominant_Topic_ID'].map(ai_topic_names)
                    st.success("AI topic naming complete!")
                except Exception as e:
                    st.warning(f"AI topic naming failed, using fallback: {e}")
                    # Fallback to basic naming
                    df['Dominant_Topic_Name'] = df['Dominant_Topic_ID'].apply(lambda x: f"Research Area {x+1}")
        else:
            # Fallback to basic naming
            df['Dominant_Topic_Name'] = df['Dominant_Topic_ID'].apply(lambda x: f"Research Area {x+1}")
        
        st.success("Topic modeling complete!")
        return df, lda_model, dictionary
def extract_topic_keywords(lda_model, dictionary, num_topics, num_words=10):
    """Extract top keywords for each topic for AI naming"""
    topic_keywords = []
    for i in range(num_topics):
        topic_terms = lda_model.get_topic_terms(i, num_words)
        keywords = [dictionary[term_id] for term_id, _ in topic_terms]
        topic_keywords.append(keywords)
    return topic_keywords

def generate_ai_topic_names(domain, lda_topics, topic_keywords):
    """Generate meaningful topic names using AI"""
    topic_keywords_str = "\n".join([f"Topic {i}: {', '.join(keywords)}" for i, keywords in enumerate(topic_keywords)])
    
    prompt = f"""
    Domain: {domain}
    
    For each topic below with its characteristic keywords, generate a concise, meaningful topic name (2-4 words) that captures the research theme:
    
    {topic_keywords_str}
    
    Provide only the topic names as a numbered list.
    """
    
    response = generate_with_groq(
        prompt,
        system_message="You are an expert at creating clear, descriptive names for technical research topics.",
        max_tokens=500
    )
    
    if response:
        # Parse the response to extract topic names
        topic_names = {}
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if '.' in line:
                name = line.split('.', 1)[1].strip()
                topic_names[i] = name
            elif i < len(lda_topics):
                topic_names[i] = line.strip()
        
        # Fill any missing topics
        for i in range(len(lda_topics)):
            if i not in topic_names:
                topic_names[i] = f"Topic {i+1}: {', '.join(topic_keywords[i][:3])}"
        
        return topic_names
    
    # Fallback to keyword-based names
    return {i: f"Topic {i+1}: {', '.join(keywords[:2])}" for i, keywords in enumerate(topic_keywords)}