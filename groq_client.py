# groq_client.py
# Groq API functionality

import streamlit as st
from groq import Groq
from config import GROQ_API_KEY, SELECTED_MODEL

def setup_groq_client():
    """Setup Groq client with API key from code"""
    if GROQ_API_KEY and GROQ_API_KEY.startswith('gsk_') and len(GROQ_API_KEY) > 20:
        return Groq(api_key=GROQ_API_KEY)
    else:
        st.sidebar.error(" Invalid Groq API key. Please update the GROQ_API_KEY variable.")
        return None

def generate_with_groq(prompt, system_message="You are a helpful AI assistant.", max_tokens=1000, temperature=0.7):
    """Generic function to generate content using Groq API"""
    client = setup_groq_client()
    if not client:
        return None
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=SELECTED_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f" Error in AI generation: {str(e)}")
        return None