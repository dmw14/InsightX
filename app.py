from bertopic import BERTopic
import streamlit as st
import pandas as pd
from gensim import corpora
import time
from gensim.models import LdaModel
from collections import Counter
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# --- NLTK Imports for the NEW clean_text function ---
from nltk.stem import WordNetLemmatizer
import string
# You need to run these downloads once in your environment for NLTK to work:
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import arxiv
import random
import re
from datetime import datetime
from groq import Groq  # Import Groq library

# --- Streamlit Setup ---
st.set_page_config(page_title="AI-Powered Tech Trend Analyzer", layout="wide")

st.title(" AI-Powered Tech Trend Analyzer")
st.write("**Fully dynamic** research analysis with AI-generated insights, topic labeling, and strategic recommendations using arXiv data.")

# =========================================================================
# === GROQ API SETUP - ENHANCED FOR DYNAMIC GENERATION ===
# =========================================================================

GROQ_API_KEY = "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"

# Current available Groq models
AVAILABLE_MODELS = {
    "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant", 
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
    "gemma2-9b-it": "gemma2-9b-it"
}

SELECTED_MODEL = "llama-3.3-70b-versatile"  # Fixed model name

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

# =========================================================================
# === AI-POWERED DYNAMIC GENERATION FUNCTIONS ===
# =========================================================================

def generate_domain_suggestions(user_input):
    """Generate relevant tech domains based on user input"""
    prompt = f"""
    Based on the user's interest in: "{user_input}"
    
    Suggest 5-8 relevant technology domains/research fields that would be appropriate for academic paper analysis.
    For each domain, provide:
    1. The domain name
    2. Corresponding arXiv category code (like cs.LG, cs.CV, etc.)
    3. 3-4 key search keywords for that domain
    
    Format as a clear list without markdown.
    """
    
    response = generate_with_groq(
        prompt,
        system_message="You are an expert in technology domains and academic research fields. Provide accurate arXiv categories and relevant keywords.",
        max_tokens=800
    )
    return response

def generate_dynamic_topic_names(domain, topic_keywords_list):
    """Generate meaningful topic names based on extracted keywords"""
    prompt = f"""
    Domain: {domain}
    
    For each set of topic keywords below, generate a concise, meaningful topic name (max 4-5 words) that captures the essence of the research area:
    
    {topic_keywords_list}
    
    Provide only the topic names in a numbered list, nothing else.
    """
    
    response = generate_with_groq(
        prompt,
        system_message="You are an expert at creating clear, descriptive names for research topics and technical domains.",
        max_tokens=500
    )
    return response

def generate_domain_keywords(domain, arxiv_category):
    """Generate relevant search keywords for a domain"""
    prompt = f"""
    Domain: {domain}
    arXiv Category: {arxiv_category}
    
    Generate 5-7 specific search keywords/phrases that would effectively find relevant research papers in this domain.
    Focus on technical terms, methodologies, and application areas.
    
    Provide only the keywords as a comma-separated list.
    """
    
    response = generate_with_groq(
        prompt,
        system_message="You are an expert at identifying relevant technical keywords for academic paper searches.",
        max_tokens=300
    )
    return response

def analyze_topic_trends(domain, topic_evolution_data):
    """Generate insights about topic trends"""
    prompt = f"""
    Domain: {domain}
    
    Analyze the following topic evolution data and provide insights:
    {topic_evolution_data}
    
    Focus on:
    1. Emerging trends (topics gaining popularity)
    2. Declining areas (topics losing interest)
    3. Stable research directions
    4. Potential interdisciplinary connections
    5. Future research opportunities
    
    Provide concise, actionable insights.
    """
    
    response = generate_with_groq(
        prompt,
        system_message="You are a research trend analyst with expertise in identifying patterns in academic literature.",
        max_tokens=1000
    )
    return response

def generate_research_gaps(domain, papers_data, future_scopes):
    """Identify research gaps and opportunities"""
    prompt = f"""
    Domain: {domain}
    
    Based on analysis of {len(papers_data)} papers and their future work sections:
    {future_scopes}
    
    Identify:
    1. Major research gaps in the field
    2. Underexplored application areas
    3. Technical challenges needing solutions
    4. Interdisciplinary opportunities
    5. High-impact research directions
    
    Provide specific, actionable research opportunities.
    """
    
    response = generate_with_groq(
        prompt,
        system_message="You are an expert research strategist skilled at identifying gaps and opportunities in academic literature.",
        max_tokens=1200
    )
    return response

# =========================================================================
# === DYNAMIC DOMAIN CONFIGURATION ===
# =========================================================================

# Default domains (fallback)
DEFAULT_DOMAINS = {
    "Machine Learning": "cs.LG",
    "Deep Learning": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Computer Vision": "cs.CV", 
    "Robotics": "cs.RO",
    "Artificial Intelligence": "cs.AI"
}

def get_dynamic_domains():
    """Get domains - either default or AI-generated based on user input"""
    st.sidebar.subheader(" Domain Selection")
    
    # Option for custom domain generation
    custom_domain = st.sidebar.text_input("Or describe your interest for AI-generated domains:", 
                                         placeholder="e.g., quantum machine learning, bioinformatics, edge AI")
    
    domains = DEFAULT_DOMAINS
    domain_keywords = {}
    
    if custom_domain and st.sidebar.button("Generate Domains with AI"):
        with st.spinner(" Generating domain suggestions..."):
            suggestions = generate_domain_suggestions(custom_domain)
            if suggestions:
                st.sidebar.success("Domain suggestions generated!")
                # Parse the response to extract domains (simplified for demo)
                # In a full implementation, you'd parse this properly
                domains = parse_domain_suggestions(suggestions)
            else:
                st.sidebar.warning("Using default domains")
    
    # Generate keywords for domains dynamically
    for domain in domains:
        if domain not in st.session_state.get('domain_keywords', {}):
            # Use regular st.spinner() instead of st.sidebar.spinner()
            keyword_status = st.sidebar.empty()
            keyword_status.info(f"Generating keywords for {domain}...")
            keywords = generate_domain_keywords(domain, domains[domain])
            keyword_status.empty()
            
            if keywords:
                domain_keywords[domain] = [k.strip() for k in keywords.split(',')]
            else:
                # Fallback keywords
                domain_keywords[domain] = [domain.lower()]
    
    st.session_state.domain_keywords = domain_keywords
    return domains, domain_keywords

def parse_domain_suggestions(suggestions):
    """Parse AI-generated domain suggestions (simplified implementation)"""
    # This is a simplified parser - in production you'd want more robust parsing
    domains = {}
    lines = suggestions.split('\n')
    for line in lines:
        if 'cs.' in line.lower():
            parts = line.split('-')
            if len(parts) >= 2:
                domain = parts[0].strip()
                category = parts[1].strip()
                domains[domain] = category
    return domains if domains else DEFAULT_DOMAINS

# =========================================================================
# === ENHANCED SCRAPING WITH AI SUPPORT ===
# =========================================================================

@st.cache_data(ttl=3600)
def fetch_historical_papers(selected_domain, arxiv_category, keywords):
    """Fetches papers with AI-enhanced keyword selection"""
    client = arxiv.Client()
    
    all_papers = []
    start_year = 2020
    end_year = 2025
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_years = end_year - start_year + 1
    
    for year_idx, year in enumerate(range(start_year, end_year + 1)):
        status_text.write(f"Fetching {selected_domain} papers from {year}... ({year_idx + 1}/{total_years})")
        
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        # Use AI-generated or fallback keywords
        search_keywords = keywords if keywords else [selected_domain.lower()]
        keyword_query = " OR ".join([f'abs:"{kw}"' for kw in search_keywords[:3]])
        
        search_query = f"cat:{arxiv_category} AND ({keyword_query}) AND submittedDate:[{start_date.strftime('%Y%m%d%H%M%S')} TO {end_date.strftime('%Y%m%d%H%M%S')}]"

        search = arxiv.Search(
            query=search_query,
            max_results=1000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending
        )

        try:
            results = list(client.results(search))
            papers_count = 0
            
            for result in results:
                if papers_count >= 1000:
                    break
                    
                abstract = result.summary
                future_scope = extract_future_scope(abstract)
                
                paper = {
                    "title": result.title,
                    "abstract": abstract,
                    "published": result.published,
                    "year": result.published.year,
                    "future_scope": future_scope,
                    "domain": selected_domain
                }
                all_papers.append(paper)
                papers_count += 1

            st.success(f"Fetched {papers_count} {selected_domain} papers from {year}")
            time.sleep(random.uniform(2, 5))

        except Exception as e:
            st.error(f"Error fetching {selected_domain} papers for {year}: {str(e)}")
            time.sleep(10)
            continue
            
        progress_bar.progress((year_idx + 1) / total_years)

    progress_bar.empty()
    status_text.empty()
    
    if all_papers:
        df = pd.DataFrame(all_papers)
        year_counts = df['year'].value_counts().sort_index()
        st.write(f"**{selected_domain} papers per year:**")
        for year, count in year_counts.items():
            st.write(f"  {year}: {count} papers")
        
        future_scope_found = df['future_scope'].notna().sum()
        st.write(f"**Future scope/conclusion found in {future_scope_found} out of {len(df)} papers ({future_scope_found/len(df)*100:.1f}%)**")
        
        return df
    else:
        st.error(f"No {selected_domain} papers were fetched")
        return pd.DataFrame()


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

def extract_topic_keywords(lda_model, dictionary, num_topics, num_words=10):
    """Extract top keywords for each topic for AI naming - CORRECTED VERSION"""
    topic_keywords = []
    for i in range(num_topics):
        topic_terms = lda_model.get_topic_terms(i, num_words)
        keywords = [dictionary[term_id] for term_id, _ in topic_terms]
        topic_keywords.append(keywords)
    return topic_keywords

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


def create_topic_distribution_table(df):
    """Create a comprehensive table showing topic distribution with counts and percentages"""
    if 'Dominant_Topic_Name' not in df.columns:
        return None
        
    topic_counts = df['Dominant_Topic_Name'].value_counts()
    total_papers = len(df)
    
    # Create summary table
    summary_data = []
    for topic, count in topic_counts.items():
        percentage = (count / total_papers) * 100
        summary_data.append({
            # 'Research Area': topic,
            'Number of Papers': count,
            'Percentage (%)': f"{percentage:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a styled table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Research Area</b>', '<b>Number of Papers</b>', '<b>Percentage (%)</b>'],
            fill_color='#2E86AB',
            align='center',
            font=dict(color='white', size=14),
            height=40
        ),
        cells=dict(
            values=[summary_df['Research Area'], summary_df['Number of Papers'], summary_df['Percentage (%)']],
            fill_color=['#F7F7F7', '#E8E8E8'],
            align=['left', 'center', 'center'],
            font=dict(size=12),
            height=35
        )
    )])
    
    fig.update_layout(
        title=' Research Areas Distribution (2020-2025 Cumulative)',
        height=400,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    return fig, summary_df

def create_topic_evolution_heatmap(df):
    """Create a heatmap showing topic evolution over years"""
    if 'Dominant_Topic_Name' not in df.columns:
        return None
        
    # Create pivot table for heatmap
    heatmap_data = df.groupby(['year', 'Dominant_Topic_Name']).size().unstack(fill_value=0)
    
    # Calculate percentages by year
    heatmap_percentage = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
    
    fig = px.imshow(
        heatmap_percentage.T,
        labels=dict(x="Year", y="Research Area", color="Percentage (%)"),
        x=heatmap_percentage.index,
        y=heatmap_percentage.columns,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Research Area Popularity Over Time (%)"
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=2020, dtick=1),
        yaxis=dict(title="Research Areas"),
        height=400
    )
    
    # Add annotations
    for i, year in enumerate(heatmap_percentage.index):
        for j, topic in enumerate(heatmap_percentage.columns):
            fig.add_annotation(
                x=year, y=topic,
                text=f"{heatmap_percentage.loc[year, topic]:.1f}%",
                showarrow=False,
                font=dict(color="white" if heatmap_percentage.loc[year, topic] > 50 else "black")
            )
    
    return fig

def create_topic_trends_with_predictions(df):
    """Create line chart showing trends from 2020-2025 and predicted trends for 2026-2027"""
    if 'Dominant_Topic_Name' not in df.columns:
        return None, None
        
    # Get actual data
    yearly_topic_counts = df.groupby(['year', 'Dominant_Topic_Name']).size().unstack(fill_value=0)
    
    # Prepare data for plotting
    plot_data = []
    predictions_data = []
    
    for topic in yearly_topic_counts.columns:
        topic_data = yearly_topic_counts[topic]
        
        # Add actual data (2020-2025)
        for year, count in topic_data.items():
            plot_data.append({
                'Year': year,
                'Research Area': topic,
                'Paper Count': count,
                'Type': 'Actual'
            })
        
        # Create predictions for 2026-2027 if we have enough data
        if len(topic_data) >= 2:
            X = np.array(topic_data.index).reshape(-1, 1)
            y = topic_data.values
            
            # Use linear regression for prediction
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for 2026 and 2027
            future_years = np.array([[2026], [2027]])
            future_predictions = model.predict(future_years)
            
            # Add predictions to data
            for year, pred in zip([2026, 2027], future_predictions):
                plot_data.append({
                    'Year': year,
                    'Research Area': topic,
                    'Paper Count': max(0, round(pred)),  # Ensure non-negative and round
                    'Type': 'Predicted'
                })
                
                predictions_data.append({
                    'Research Area': topic,
                    'Year': year,
                    'Predicted Papers': max(0, round(pred))
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the line chart
    fig = px.line(
        plot_df,
        x='Year',
        y='Paper Count',
        color='Research Area',
        line_dash='Type',
        title='Research Area Trends & Predictions (2020-2027)',
        labels={'Paper Count': 'Number of Papers', 'Year': 'Year'},
        markers=True
    )
    
    # Customize the appearance
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=2020,
            dtick=1,
            title="Year"
        ),
        yaxis=dict(title="Number of Papers"),
        legend=dict(
            title="Research Areas",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        ),
        height=500
    )
    
    # Style actual vs predicted lines
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )
    
    # Add value annotations for predicted years
    predicted_data = plot_df[plot_df['Type'] == 'Predicted']
    for _, row in predicted_data.iterrows():
        fig.add_annotation(
            x=row['Year'],
            y=row['Paper Count'],
            text=str(int(row['Paper Count'])),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            bgcolor="white",
            bordercolor="black"
        )
    
    predictions_df = pd.DataFrame(predictions_data) if predictions_data else None
    
    return fig, predictions_df

def create_future_scope_analysis(df):
    """Analyze and visualize future scope mentions across topics"""
    # Count future scope mentions by topic
    if 'Dominant_Topic_Name' not in df.columns:
        return None
        
    future_scope_by_topic = df.groupby('Dominant_Topic_Name').apply(
        lambda x: x['future_scope'].notna().sum()
    ).reset_index(name='future_scope_count')
    
    total_by_topic = df['Dominant_Topic_Name'].value_counts().reset_index()
    total_by_topic.columns = ['Dominant_Topic_Name', 'total_papers']
    
    # Merge and calculate percentages
    analysis_df = pd.merge(future_scope_by_topic, total_by_topic, on='Dominant_Topic_Name')
    analysis_df['future_scope_percentage'] = (analysis_df['future_scope_count'] / analysis_df['total_papers']) * 100
    analysis_df['future_scope_percentage_str'] = analysis_df['future_scope_percentage'].apply(lambda x: f"{x:.1f}%")
    
    # Create visualization
    fig = px.bar(
        analysis_df,
        x='Dominant_Topic_Name',
        y='future_scope_percentage',
        title=' Future Research Directions by Area',
        labels={
            'Dominant_Topic_Name': 'Research Area',
            'future_scope_percentage': 'Papers with Future Scope (%)'
        },
        text='future_scope_percentage_str',
        color='future_scope_percentage',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False,
        yaxis=dict(title='Papers with Future Scope (%)')
    )
    
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside'
    )
    
    return fig, analysis_df

def create_research_maturity_chart(df):
    """Create a radar chart showing research maturity across topics"""
    if 'Dominant_Topic_Name' not in df.columns:
        return None, None
        
    # Calculate maturity metrics for each topic
    topic_metrics = []
    
    for topic in df['Dominant_Topic_Name'].unique():
        topic_data = df[df['Dominant_Topic_Name'] == topic]
        
        # Metrics
        paper_count = len(topic_data)
        years_active = topic_data['year'].nunique()
        future_scope_ratio = topic_data['future_scope'].notna().mean()
        year_span = topic_data['year'].max() - topic_data['year'].min() + 1
        
        # Normalize metrics (0-100 scale)
        max_papers = df['Dominant_Topic_Name'].value_counts().max()
        paper_score = (paper_count / max_papers) * 100
        
        max_years = df['year'].nunique()
        longevity_score = (years_active / max_years) * 100
        
        future_score = future_scope_ratio * 100
        
        consistency_score = (year_span / max_years) * 100
        
        topic_metrics.append({
            'Research Area': topic,
            'Paper Volume': paper_score,
            'Longevity': longevity_score,
            'Future Focus': future_score,
            'Consistency': consistency_score
        })
    
    metrics_df = pd.DataFrame(topic_metrics)
    
    # Create radar chart for first topic as example
    if not metrics_df.empty:
        sample_topic = metrics_df.iloc[0]
        
        categories = ['Paper Volume', 'Longevity', 'Future Focus', 'Consistency']
        values = [sample_topic[cat] for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the radar chart
            theta=categories + [categories[0]],
            fill='toself',
            name=sample_topic['Research Area'],
            line=dict(color='#2E86AB')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title=f"📡 Research Maturity: {sample_topic['Research Area']}",
            height=400
        )
        
        return fig, metrics_df
    
    return None, None


def create_topic_wordclouds(df, lda_model, dictionary, num_words=20):
    """Create word clouds for each topic"""
    if lda_model is None or dictionary is None:
        return None
    
    # Extract topic keywords
    topic_keywords = []
    for i in range(lda_model.num_topics):
        topic_terms = lda_model.get_topic_terms(i, num_words)
        keywords = [dictionary[term_id] for term_id, _ in topic_terms]
        topic_keywords.append(keywords)
    
    # Create word clouds
    figs = []
    for i, keywords in enumerate(topic_keywords):
        # Create frequency dictionary for word cloud
        word_freq = {word: len(keywords) - j for j, word in enumerate(keywords)}
        
        # Create word cloud
        wordcloud = WordCloud(
            width=400, 
            height=300, 
            background_color='white',
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        
        # Convert to plotly figure
        fig = px.imshow(wordcloud, title=f"☁️ Topic {i+1} Keywords")
        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=350
        )
        figs.append(fig)
    
    return figs

def create_topic_network_analysis(df, lda_model, dictionary, num_words=15):
    """Create a network visualization showing topic relationships"""
    if lda_model is None or dictionary is None:
        return None
    
    # Extract topic keywords and create similarity matrix
    num_topics = lda_model.num_topics
    topic_keywords = []
    
    for i in range(num_topics):
        topic_terms = lda_model.get_topic_terms(i, num_words)
        keywords = [dictionary[term_id] for term_id, _ in topic_terms]
        topic_keywords.append(keywords)
    
    # Calculate topic similarity based on shared keywords
    similarity_matrix = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                set_i = set(topic_keywords[i][:10])  # Use top 10 keywords for similarity
                set_j = set(topic_keywords[j][:10])
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                similarity_matrix[i][j] = intersection / union if union > 0 else 0
    
    # Create network graph
    edge_x = []
    edge_y = []
    edge_weights = []
    
    node_x = []
    node_y = []
    node_text = []
    
    # Position nodes in a circle
    radius = 1
    for i in range(num_topics):
        angle = 2 * np.pi * i / num_topics
        node_x.append(radius * np.cos(angle))
        node_y.append(radius * np.sin(angle))
        node_text.append(f"Topic {i+1}")
    
    # Create edges for significant similarities
    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            if similarity_matrix[i][j] > 0.1:  # Only show significant connections
                edge_x.extend([node_x[i], node_x[j], None])
                edge_y.extend([node_y[i], node_y[j], None])
                edge_weights.append(similarity_matrix[i][j])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=30,
            color='#2E86AB',
            line=dict(width=2, color='white')
        ))
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Topic Relationship Network',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       annotations=[dict(
                           text="Node size represents topic importance<br>Line thickness shows keyword similarity",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002)],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))
    
    return fig, similarity_matrix


if 'papers' not in st.session_state:
    st.session_state.papers = None
if 'current_domain' not in st.session_state:
    st.session_state.current_domain = None
if 'domain_keywords' not in st.session_state:
    st.session_state.domain_keywords = {}
if 'lda_model' not in st.session_state:
    st.session_state.lda_model = None
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = None

# --- Sidebar Configuration ---
st.sidebar.header(" AI-Powered Configuration")

# Get dynamic domains
tech_domains, domain_keywords = get_dynamic_domains()

selected_domain = st.sidebar.selectbox(
    "Select Tech Domain:",
    options=list(tech_domains.keys()),
    index=0
)

arxiv_category = tech_domains[selected_domain]
keywords = domain_keywords.get(selected_domain, [selected_domain.lower()])

st.sidebar.info(f"**Selected:** {selected_domain}  \n**arXiv:** {arxiv_category}  \n**Keywords:** {', '.join(keywords[:3])}")

# --- Groq API Status ---
st.sidebar.header("AI Features")
groq_client = setup_groq_client()
if groq_client:
    st.sidebar.success(f"Groq API connected! Using: {SELECTED_MODEL}")
else:
    st.sidebar.error("Groq API not configured")

enable_ai_features = st.sidebar.checkbox("Enable AI-Powered Features", value=True,
                                        help="Use AI for dynamic topic naming, insights, and recommendations")

# --- Data Fetching ---
if st.sidebar.button(f"Fetch {selected_domain} Papers (2020-2025)"):
    try:
        stopwords.words('english')
    except LookupError:
        st.error("NLTK resources missing! Please install NLTK and run the required downloads.")
        st.stop()
        
    try:
        with st.spinner(f"Fetching {selected_domain} papers from arXiv (2020-2025)…"):
            df = fetch_historical_papers(selected_domain, arxiv_category, keywords)
            st.session_state.papers = df
            st.session_state.current_domain = selected_domain
    except Exception as e:
        st.error(f"Error fetching {selected_domain} data: {e}")
        st.session_state.papers = None

# Display current domain if papers exist
if st.session_state.papers is not None and not st.session_state.papers.empty:
    current_domain = st.session_state.current_domain
    st.header(f"{current_domain} Analysis Results")
    df = st.session_state.papers

    # --- Quick Stats Overview ---
    st.subheader("Quick Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_papers = len(df)
        st.metric("Total Papers", total_papers)
    
    with col2:
        years_covered = df['year'].nunique()
        st.metric("Years Covered", years_covered)
    
    with col3:
        future_scope_papers = df['future_scope'].notna().sum()
        st.metric("Papers with Future Scope", f"{future_scope_papers} ({future_scope_papers/total_papers*100:.1f}%)")
    
    with col4:
        topics_discovered = df['Dominant_Topic_Name'].nunique() if 'Dominant_Topic_Name' in df.columns else 0
        st.metric("Research Areas Found", topics_discovered)

    # --- AI-Enhanced Analysis Pipeline ---
    st.header("1. AI-Powered Topic Modeling")
    
    with st.spinner("Preprocessing and modeling with AI enhancements..."):
        publication_years = df['year'].tolist()
        abstracts = df['abstract'].tolist()
        cleaned_abstracts = [clean_text(abstract) for abstract in abstracts]
        
        # LDA Setup
        tokenized_texts = [text.split() for text in cleaned_abstracts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # LDA Model
        num_topics_lda = 5
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics_lda, passes=15, random_state=42)
        dominant_topics = [max(lda_model.get_document_topics(doc_bow), key=lambda x: x[1])[0] for doc_bow in corpus]
        df['Dominant_Topic_ID'] = dominant_topics
        
        # Store models in session state for later use
        st.session_state.lda_model = lda_model
        st.session_state.dictionary = dictionary
        
        # AI-Powered Topic Naming
        if enable_ai_features and groq_client:
            with st.spinner("Generating AI-powered topic names..."):
                topic_keywords = extract_topic_keywords(lda_model, dictionary, num_topics_lda, num_words=10)
                ai_topic_names = generate_ai_topic_names(current_domain, range(num_topics_lda), topic_keywords)
                df['Dominant_Topic_Name'] = df['Dominant_Topic_ID'].map(ai_topic_names)
                st.success("AI topic naming complete!")
        else:
            # Fallback to basic naming
            df['Dominant_Topic_Name'] = df['Dominant_Topic_ID'].apply(lambda x: f"Topic {x+1}")
        
        st.success("Topic modeling complete!")

    st.header("2. Interactive Research Dashboard")
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", 
        "Trends", 
        "Future Research", 
        "Maturity", 
        "Keywords", 
        "Relationships"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.subheader("Research Areas Overview")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            dist_table, dist_df = create_topic_distribution_table(df)
            if dist_table:
                st.plotly_chart(dist_table, use_container_width=True)
            else:
                st.info("Topic distribution data not available")
        
        with col2:
            heatmap = create_topic_evolution_heatmap(df)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
            else:
                st.info("Topic evolution data not available")
        
        # Show detailed statistics
        if dist_df is not None:
            with st.expander("Detailed Research Area Statistics"):
                st.dataframe(dist_df, use_container_width=True)
    
    # Tab 2: Trends
    with tab2:
        st.subheader("Research Trends & Predictions")
        
        trends_chart, predictions_df = create_topic_trends_with_predictions(df)
        if trends_chart:
            st.plotly_chart(trends_chart, use_container_width=True)
            
            if predictions_df is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    with st.expander("View Predictions Table"):
                        st.write("**Predicted Paper Counts for 2026-2027:**")
                        pivot_df = predictions_df.pivot(index='Research Area', columns='Year', values='Predicted Papers')
                        st.dataframe(pivot_df, use_container_width=True)
                
                with col2:
                    st.metric("Total Predicted Papers (2026-2027)", 
                             int(predictions_df['Predicted Papers'].sum()))
        else:
            st.info("Trend analysis data not available")
    
    # Tab 3: Future Research
    with tab3:
        st.subheader("Future Research Directions Analysis")
        
        future_chart, future_df = create_future_scope_analysis(df)
        if future_chart:
            st.plotly_chart(future_chart, use_container_width=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.expander("Future Scope Details"):
                    if future_df is not None:
                        st.dataframe(future_df[['Dominant_Topic_Name', 'future_scope_count', 'total_papers', 'future_scope_percentage']], 
                                   use_container_width=True)
            
            with col2:
                if future_df is not None:
                    avg_future_scope = future_df['future_scope_percentage'].mean()
                    max_future_scope = future_df['future_scope_percentage'].max()
                    
                    st.metric("Average Future Scope", f"{avg_future_scope:.1f}%")
                    st.metric("Highest Future Focus", f"{max_future_scope:.1f}%")
        else:
            st.info("Future scope analysis data not available")
    
    # Tab 4: Research Maturity
    with tab4:
        st.subheader("Research Maturity Analysis")
        
        maturity_chart, maturity_df = create_research_maturity_chart(df)
        if maturity_chart:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(maturity_chart, use_container_width=True)
            
            with col2:
                st.write("**Research Maturity Metrics:**")
                st.write("• **Paper Volume**: Total research output")
                st.write("• **Longevity**: Years of active research")
                st.write("• **Future Focus**: Emphasis on future directions")
                st.write("• **Consistency**: Steady research presence")
                
                if maturity_df is not None:
                    # Show maturity scores for all topics
                    with st.expander("View All Topics Maturity Scores"):
                        display_df = maturity_df.copy()
                        for col in ['Paper Volume', 'Longevity', 'Future Focus', 'Consistency']:
                            display_df[col] = display_df[col].round(1)
                        st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Research maturity analysis data not available")
    
    # Tab 5: Keyword Word Clouds
    with tab5:
        st.subheader("Topic Keyword Visualization")
        
        wordcloud_figs = create_topic_wordclouds(df, st.session_state.lda_model, st.session_state.dictionary)
        
        if wordcloud_figs:
            # Display word clouds in columns
            cols = st.columns(2)
            for i, fig in enumerate(wordcloud_figs):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show topic keywords as text
                    if 'Dominant_Topic_Name' in df.columns:
                        topic_name = df[df['Dominant_Topic_ID'] == i]['Dominant_Topic_Name'].iloc[0]
                        st.caption(f"**{topic_name}** - Top keywords visualized")
        else:
            st.info("Word cloud data not available")
    
    # Tab 6: Topic Relationships
    with tab6:
        st.subheader("Topic Relationship Network")
        
        network_fig, similarity_matrix = create_topic_network_analysis(
            df, st.session_state.lda_model, st.session_state.dictionary
        )
        
        if network_fig:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.plotly_chart(network_fig, use_container_width=True)
            
            with col2:
                st.write("**Network Insights:**")
                st.write("• **Nodes**: Research topics")
                st.write("• **Connections**: Shared keywords")
                st.write("• **Thickness**: Similarity strength")
                
                if similarity_matrix is not None:
                    with st.expander("View Similarity Matrix"):
                        # Create a readable similarity matrix
                        sim_df = pd.DataFrame(
                            similarity_matrix,
                            index=[f"Topic {i+1}" for i in range(len(similarity_matrix))],
                            columns=[f"Topic {i+1}" for i in range(len(similarity_matrix))]
                        )
                        st.dataframe(sim_df.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("Network analysis data not available")

    # --- Sample Papers Display ---
    st.header("3. Sample Research Papers")
    
    if 'Dominant_Topic_Name' in df.columns:
        topic_select = st.selectbox("Select Research Area to View Papers:", df['Dominant_Topic_Name'].unique())
        
        topic_papers = df[df['Dominant_Topic_Name'] == topic_select].head(5)
        
        st.write(f"**Showing 5 sample papers from: {topic_select}**")
        
        for idx, paper in topic_papers.iterrows():
            with st.expander(f"📖 {paper['title']}"):
                st.write(f"**Published:** {paper['year']}")
                st.write(f"**Abstract:** {paper['abstract'][:400]}...")
                if paper['future_scope']:
                    st.write(f"**Future Research Directions:** {paper['future_scope'][:300]}...")
                else:
                    st.write("**Future Research Directions:** Not specified in abstract")
    else:
        st.info("Topic information not available")

    # --- AI-Powered Analysis Section ---
    if enable_ai_features and groq_client:
        st.header("4. AI Research Insights")
        
        # Research Gaps Analysis
        with st.spinner(" Analyzing research gaps and opportunities..."):
            future_scopes = df[df['future_scope'].notna()]['future_scope'].tolist()[:10]
            if future_scopes:
                research_gaps = generate_research_gaps(current_domain, df, future_scopes)
                if research_gaps:
                    st.subheader(" AI-Identified Research Opportunities")
                    st.markdown(research_gaps)
            else:
                st.info("No future scope sections found for analysis.")

        # Topic Trend Analysis
        with st.spinner(" Analyzing topic evolution trends..."):
            topic_trends = df.groupby(['year', 'Dominant_Topic_Name']).size().reset_index()
            trend_analysis = analyze_topic_trends(current_domain, topic_trends.to_string())
            if trend_analysis:
                st.subheader(" AI Topic Trend Analysis")
                st.markdown(trend_analysis)

    # --- Data Export ---
    st.header("5.  Export Results")
    
    if st.button(" Download Complete Analysis"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name=f"{current_domain.replace(' ', '_')}_analysis.csv",
            mime="text/csv"
        )

# Add instructions for first-time users
else:
    st.info("""
    ## 🚀 Welcome to AI-Powered Tech Trend Analyzer!
    
    **To get started:**
    1. Select a tech domain from the sidebar
    2. Or describe your interest for AI-generated domain suggestions
    3. Click "Fetch Papers" to begin analysis
    4. Enable AI features for dynamic insights and topic naming
    
    **What you'll discover:**
    -  Research area distribution and percentages
    -  Topic popularity evolution over time
    -  Trends from 2020-2025 and predictions for 2026-2027
    -  Future research directions by topic
    -  Research maturity analysis
    -  Keyword visualizations
    -  Topic relationship networks
    
    **Perfect for:**
    - Researchers exploring new areas
    - Students finding thesis topics
    - Companies identifying innovation opportunities
    - Investors tracking tech trends
    """)