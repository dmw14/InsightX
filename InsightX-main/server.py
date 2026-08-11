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
st.set_page_config(page_title="Multi-Domain Tech Trend Analyzer", layout="wide")

st.title("🤖 Multi-Domain Tech Trend Analyzer")
st.write("Advanced research analysis with **AI-powered insights** for multiple tech domains using arXiv data (2020-2025).")

# =========================================================================
# === GROQ API SETUP - SIMPLIFIED ===
# =========================================================================

# Add your Groq API key directly here
GROQ_API_KEY = "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"  # ← REPLACE WITH YOUR ACTUAL API KEY

# Current available Groq models (December 2024)
AVAILABLE_MODELS = {
    "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant", 
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
    "gemma2-9b-it": "gemma2-9b-it"
}

SELECTED_MODEL = "llama-3.3-70b-versatile"  # Current best model

def setup_groq_client():
    """Setup Groq client with API key from code"""
    if GROQ_API_KEY and GROQ_API_KEY.startswith('gsk_') and len(GROQ_API_KEY) > 20:
        return Groq(api_key=GROQ_API_KEY)
    else:
        st.sidebar.error("❌ Invalid Groq API key. Please update the GROQ_API_KEY variable in the code.")
        return None

def generate_ai_insights(domain, top_topics, forecast_data, future_scope_data, trend_analysis):
    """Generate AI-powered insights using Groq API"""
    client = setup_groq_client()
    if not client:
        return "⚠️ Groq API key not configured. Please update the GROQ_API_KEY variable in the code."
    
    try:
        # Prepare context for the AI
        context = f"""
        Domain: {domain}
        Top 3 High-Growth Topics: {', '.join(top_topics)}
        Forecast Data: {forecast_data}
        Future Scope Analysis: {future_scope_data}
        Trend Analysis: {trend_analysis}
        
        Please provide strategic recommendations for research and development focusing on:
        1. Specific research directions based on the forecasted high-growth areas
        2. Practical applications and commercialization opportunities
        3. Potential risks and challenges to consider
        4. Emerging sub-fields worth monitoring
        5. Collaboration opportunities with academia/industry
        
        Be concise, actionable, and data-driven in your recommendations.
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic technology advisor with deep expertise in research trend analysis and technology forecasting. Provide concise, actionable insights based on the data provided. Focus on practical recommendations that can guide R&D investment decisions."
                },
                {
                    "role": "user",
                    "content": f"Based on the following research analysis, provide strategic recommendations:\n{context}"
                }
            ],
            model=SELECTED_MODEL,
            temperature=0.7,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error generating AI insights: {str(e)}"

def generate_topic_specific_recommendations(domain, topic_name, future_scopes, forecast_growth):
    """Generate AI-powered recommendations for specific topics"""
    client = setup_groq_client()
    if not client:
        return "⚠️ Groq API key not configured."
    
    try:
        # Sample some future scope content for context
        future_scope_samples = future_scopes[:3] if future_scopes else ["No specific future scope mentioned in papers"]
        
        context = f"""
        Domain: {domain}
        Specific Topic: {topic_name}
        Forecasted Growth: {forecast_growth}
        Future Research Directions from Papers: {future_scope_samples}
        
        Provide focused recommendations for this specific research area including:
        - Immediate research opportunities and low-hanging fruit
        - Technical challenges that need to be addressed
        - Potential real-world applications and market opportunities
        - Required skills and expertise for this domain
        - Key papers or researchers to follow in this area
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a domain expert providing specific, actionable research guidance for academic and industrial R&D teams. Focus on practical, implementable advice that can be acted upon immediately."
                },
                {
                    "role": "user",
                    "content": f"Provide specific recommendations for this research topic:\n{context}"
                }
            ],
            model=SELECTED_MODEL,
            temperature=0.7,
            max_tokens=800
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error generating topic recommendations: {str(e)}"

# =========================================================================
# === DOMAIN CONFIGURATION ===
# =========================================================================

TECH_DOMAINS = {
    "Machine Learning": "cs.LG",
    "Deep Learning": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Computer Vision": "cs.CV", 
    "Robotics": "cs.RO",
    "Artificial Intelligence": "cs.AI",
    "Data Science": "cs.DS",
    "Human-Computer Interaction": "cs.HC",
    "Computer Networks": "cs.NI",
    "Software Engineering": "cs.SE",
    "Cryptography": "cs.CR",
    "Theoretical Computer Science": "cs.CC"
}

DOMAIN_KEYWORDS = {
    "Machine Learning": ["machine learning", "ml", "supervised learning", "unsupervised learning"],
    "Deep Learning": ["deep learning", "neural network", "cnn", "rnn", "transformer", "bert"],
    "Natural Language Processing": ["natural language", "nlp", "text mining", "sentiment analysis", "language model"],
    "Computer Vision": ["computer vision", "image processing", "object detection", "segmentation"],
    "Robotics": ["robotics", "robot", "autonomous", "motion planning"],
    "Artificial Intelligence": ["artificial intelligence", "ai", "intelligent system"],
    "Data Science": ["data science", "data mining", "big data", "analytics"],
    "Human-Computer Interaction": ["hci", "user interface", "ux", "human computer"],
    "Computer Networks": ["computer network", "wireless", "5g", "network security"],
    "Software Engineering": ["software engineering", "agile", "devops", "code quality"],
    "Cryptography": ["cryptography", "encryption", "security protocol", "blockchain"],
    "Theoretical Computer Science": ["algorithm", "complexity", "computation", "theory"]
}

# =========================================================================
# === UPDATED SCRAPING FUNCTION ===
# =========================================================================

@st.cache_data(ttl=3600)
def fetch_historical_papers(selected_domain, arxiv_category):
    """Fetches exactly 100 papers from each year (2020-2025) with future scope extraction for selected domain."""
    client = arxiv.Client()
    
    all_papers = []
    start_year = 2020
    end_year = 2025
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_years = end_year - start_year + 1
    
    for year_idx, year in enumerate(range(start_year, end_year + 1)):
        status_text.write(f"Fetching {selected_domain} papers from {year}... ({year_idx + 1}/{total_years})")
        
        # Define date range for the entire year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        # Use both arXiv category and keyword filtering for better domain specificity
        domain_keywords = DOMAIN_KEYWORDS.get(selected_domain, [selected_domain.lower()])
        keyword_query = " OR ".join([f'abs:"{kw}"' for kw in domain_keywords[:3]])
        
        search_query = f"cat:{arxiv_category} AND ({keyword_query}) AND submittedDate:[{start_date.strftime('%Y%m%d%H%M%S')} TO {end_date.strftime('%Y%m%d%H%M%S')}]"

        search = arxiv.Search(
            query=search_query,
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending
        )

        try:
            results = list(client.results(search))
            papers_count = 0
            
            for result in results:
                if papers_count >= 100:
                    break
                    
                # Extract future scope or conclusion from abstract
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

            st.success(f"✅ Fetched {papers_count} {selected_domain} papers from {year}")
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
        st.write(f"**📊 Future scope/conclusion found in {future_scope_found} out of {len(df)} papers ({future_scope_found/len(df)*100:.1f}%)**")
        
        return df
    else:
        st.error(f"❌ No {selected_domain} papers were fetched")
        return pd.DataFrame()

def extract_future_scope(abstract):
    """
    Extract future scope or conclusion section from abstract using pattern matching.
    Returns the extracted text or None if not found.
    """
    if not abstract:
        return None
    
    # Convert to lowercase for case-insensitive matching
    text_lower = abstract.lower()
    
    # Patterns that indicate future work or conclusion sections
    future_patterns = [
        r'(future work.*?)(?=\n\n|\n[A-Z]|$)',
        r'(future research.*?)(?=\n\n|\n[A-Z]|$)',
        r'(future directions.*?)(?=\n\n|\n[A-Z]|$)',
        r'(limitations and future work.*?)(?=\n\n|\n[A-Z]|$)',
        r'(conclusion and future work.*?)(?=\n\n|\n[A-Z]|$)',
        r'(we conclude.*?)(?=\n\n|\n[A-Z]|$)',
        r'(in conclusion.*?)(?=\n\n|\n[A-Z]|$)',
        r'(to conclude.*?)(?=\n\n|\n[A-Z]|$)',
        r'(this paper concludes.*?)(?=\n\n|\n[A-Z]|$)',
        r'(our work concludes.*?)(?=\n\n|\n[A-Z]|$)',
    ]
    
    conclusion_patterns = [
        r'(conclusion.*?)(?=\n\n|\n[A-Z]|$)',
        r'(discussion and conclusion.*?)(?=\n\n|\n[A-Z]|$)',
        r'(summary and conclusion.*?)(?=\n\n|\n[A-Z]|$)',
    ]
    
    # First try to find future work sections
    for pattern in future_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            # Get the original case text from abstract
            start_pos = abstract.lower().find(match.group(1))
            if start_pos != -1:
                end_pos = start_pos + len(match.group(1))
                return abstract[start_pos:end_pos].strip()
    
    # If no future work found, try conclusion sections
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            # Get the original case text from abstract
            start_pos = abstract.lower().find(match.group(1))
            if start_pos != -1:
                end_pos = start_pos + len(match.group(1))
                return abstract[start_pos:end_pos].strip()
    
    return None

@st.cache_data
def clean_text(text):
    """Performs comprehensive text cleaning, stopword removal, and lemmatization."""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    domain_stopwords = {'algorithm', 'method', 'problem', 'model', 'data', 'result', 'approach', 'paper','user'}
    stop_words.update(domain_stopwords)

    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into one string
    cleaned_text = ' '.join(words)

    return cleaned_text

# =========================================================================
# =========================================================================

# --- Domain Selection ---
st.sidebar.header("🔧 Analysis Configuration")

selected_domain = st.sidebar.selectbox(
    "Select Tech Domain:",
    options=list(TECH_DOMAINS.keys()),
    index=0,
    help="Choose the technology domain you want to analyze"
)

arxiv_category = TECH_DOMAINS[selected_domain]

st.sidebar.info(f"**Selected Domain:** {selected_domain}  \n**arXiv Category:** {arxiv_category}")

# --- Groq API Status ---
st.sidebar.header("🤖 AI Insights Configuration")
groq_client = setup_groq_client()
if groq_client:
    st.sidebar.success(f"✅ Groq API connected! Using: {SELECTED_MODEL}")
else:
    st.sidebar.error("❌ Groq API not configured")

enable_ai_insights = st.sidebar.checkbox("Enable AI-Powered Insights", value=True, 
                                        help="Generate dynamic AI recommendations using Groq")

# --- Data Fetching ---
col1, col2 = st.columns([1, 4])
if col1.button(f"Fetch {selected_domain} Papers (2020-2025)"):
    # Ensure NLTK resources are available
    try:
        stopwords.words('english')
    except LookupError:
        st.error("NLTK resources missing! Please install NLTK and run `nltk.download('punkt'), nltk.download('stopwords'), nltk.download('wordnet')` in your environment.")
        st.stop()
        
    try:
        with st.spinner(f"Fetching {selected_domain} papers from arXiv (2020-2025)… This may take a few minutes…"):
            df = fetch_historical_papers(selected_domain, arxiv_category)
            st.session_state['papers'] = df
            st.session_state['current_domain'] = selected_domain
    except Exception as e:
        st.error(f"Error fetching {selected_domain} data: {e}. Check your network connection and the `arxiv` library.")
        df = pd.DataFrame()
        st.session_state['papers'] = df

    if not df.empty:
        st.success(f"Successfully fetched **{len(df)}** {selected_domain} papers from 2020-2025!")
        st.dataframe(df[['title', 'year', 'future_scope']].head())
        
        # Show future scope preview
        if df['future_scope'].notna().sum() > 0:
            st.subheader("Future Scope Examples")
            future_scope_samples = df[df['future_scope'].notna()]['future_scope'].head(3).tolist()
            for i, scope in enumerate(future_scope_samples):
                st.write(f"**Example {i+1}:** {scope[:200]}..." if len(scope) > 200 else f"**Example {i+1}:** {scope}")
    else:
        st.error(f"Could not fetch {selected_domain} papers.")

# Display current domain if papers exist
if "papers" in st.session_state and not st.session_state['papers'].empty:
    current_domain = st.session_state.get('current_domain', 'Unknown Domain')
    st.header(f"📊 {current_domain} Analysis Results")

# ------------------ Process Only If Data Exists ------------------
if "papers" in st.session_state and not st.session_state['papers'].empty:
    df = st.session_state['papers']
    current_domain = st.session_state.get('current_domain', 'Selected Domain')
    df['year'] = df['published'].dt.year # Ensure 'year' is consistent

    # --- Domain-specific topic names ---
    DOMAIN_TOPIC_NAMES = {
        "Machine Learning": {
            0: "Neural Networks & Deep Learning",
            1: "Reinforcement Learning", 
            2: "Probabilistic Models & Uncertainty",
            3: "Optimization Methods & Efficiency",
            4: "Supervised Learning & Classification"
        },
        "Deep Learning": {
            0: "Computer Vision & CNNs",
            1: "Natural Language Processing & Transformers",
            2: "Generative Models & GANs",
            3: "Recurrent Networks & Sequential Data",
            4: "Deep Reinforcement Learning"
        },
        "Natural Language Processing": {
            0: "Language Models & Transformers",
            1: "Text Classification & Sentiment Analysis",
            2: "Machine Translation & Multilingual NLP",
            3: "Information Extraction & NER",
            4: "Dialogue Systems & Chatbots"
        },
        "Computer Vision": {
            0: "Object Detection & Recognition",
            1: "Image Segmentation & Processing",
            2: "Video Analysis & Action Recognition",
            3: "3D Vision & Reconstruction",
            4: "Biometrics & Facial Recognition"
        },
        "Robotics": {
            0: "Motion Planning & Control",
            1: "Robot Learning & Manipulation",
            2: "Autonomous Navigation",
            3: "Human-Robot Interaction",
            4: "Swarm Robotics & Multi-agent Systems"
        },
        "Artificial Intelligence": {
            0: "Machine Learning Applications",
            1: "Knowledge Representation & Reasoning",
            2: "Planning & Scheduling",
            3: "AI Ethics & Fairness",
            4: "Multi-agent Systems"
        },
        "Data Science": {
            0: "Data Mining & Pattern Discovery",
            1: "Big Data Analytics & Processing",
            2: "Statistical Modeling & Inference",
            3: "Data Visualization & Exploration",
            4: "Predictive Analytics & Forecasting"
        },
        "Human-Computer Interaction": {
            0: "User Interface Design",
            1: "Usability Testing & Evaluation",
            2: "Accessibility & Inclusive Design",
            3: "Mobile & Ubiquitous Computing",
            4: "Social Computing & CSCW"
        }
    }
    
    # Default topic names for domains not specified
    default_topic_names = {
        0: "Topic 1: Core Research Area",
        1: "Topic 2: Methodological Advances", 
        2: "Topic 3: Applications & Use Cases",
        3: "Topic 4: Theoretical Foundations",
        4: "Topic 5: Emerging Trends"
    }
    
    # Get domain-specific topic names or use default
    lda_topic_names = DOMAIN_TOPIC_NAMES.get(current_domain, default_topic_names)

    # --- Preprocessing & LDA ---
    st.header("1. Topic Modeling (LDA & BERTopic)")
    st.subheader("Preprocessing Data")
    
    with st.spinner("Preprocessing text and preparing models..."):
        publication_years = df['year'].tolist()
        abstracts = df['abstract'].tolist()
        # Apply the new NLTK-based clean_text function
        cleaned_abstracts = [clean_text(abstract) for abstract in abstracts]
        
        # LDA Setup
        tokenized_texts = [text.split() for text in cleaned_abstracts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # --- LDA Model ---
        num_topics_lda = 5
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics_lda, passes=15, random_state=42)
        dominant_topics = [max(lda_model.get_document_topics(doc_bow), key=lambda x: x[1])[0] for doc_bow in corpus]
        df['Dominant_Topic_ID'] = dominant_topics
        df['Dominant_Topic_Name'] = df['Dominant_Topic_ID'].map(lda_topic_names)
        
        st.success("Preprocessing and LDA topic assignment complete.")

    # --- Cluster Distribution Table ---
    st.subheader("Topic Cluster Distribution")
    cluster_counts = df['Dominant_Topic_Name'].value_counts().reset_index()
    cluster_counts.columns = ['Topic Cluster', 'Number of Papers']
    cluster_counts['Percentage'] = (cluster_counts['Number of Papers'] / len(df) * 100).round(2)
    
    # Display the table
    st.dataframe(cluster_counts.style.format({'Percentage': '{:.2f}%'}))

    st.subheader("LDA Topic Distribution")
    topic_counts_lda = df['Dominant_Topic_Name'].value_counts().reset_index()
    topic_counts_lda.columns = ['Topic Name', 'Number of Papers']
    
    # Plotting LDA Distribution
    fig_lda_bar = px.bar(
        topic_counts_lda, 
        x='Topic Name', 
        y='Number of Papers', 
        title=f'{current_domain} - Distribution of Papers Across LDA Topics',
        color='Number of Papers'
    )
    st.plotly_chart(fig_lda_bar, use_container_width=True)

    # --- BERTopic Model ---
    st.subheader("BERTopic Analysis (Advanced Topic Modeling)")
    with st.spinner("Running BERTopic for dense topic modeling... This may take a moment."):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # BERTopic uses UMAP and HDBSCAN, which can take time
        topic_model = BERTopic(embedding_model=embedding_model, verbose=False, n_gram_range=(1, 2))
        
        topics_bertopic, probs = topic_model.fit_transform(cleaned_abstracts)
        
        topic_info = topic_model.get_topic_info()
        bertopic_topic_names = {
            row['Topic']: row['Name'] for _, row in topic_info.iterrows()
            if row['Topic'] != -1
        }
        st.success(f"BERTopic finished! Found {len(bertopic_topic_names)} main topics.")
        st.dataframe(topic_info.head(5).style.set_caption("Top BERTopics (Topic -1 is Outlier)"))
        
    st.subheader("BERTopic Trends Over Time")
    
    with st.spinner("Generating BERTopic trend visualization..."):
        # Ensure 'year' is used as the timestamp
        # FIXED: Changed from 'topos_over_time' to 'topics_over_time'
        topics_over_time = topic_model.topics_over_time(cleaned_abstracts, timestamps=publication_years, nr_bins=min(len(df['year'].unique()), 15))
        
        # Convert BERTopic Plotly figure to Streamlit display
        fig_bertopic_trends = topic_model.visualize_topics_over_time(
            topics_over_time,
            custom_labels=[bertopic_topic_names.get(t, f"Topic {t}") for t in topics_over_time.Topic.unique()]
        )
        fig_bertopic_trends.update_layout(
            title=f"{current_domain} - Research Trends Over Time",
            xaxis_title="Year",
            yaxis_title="Topic Frequency (Normalized)",
            legend_title="Research Area"
        )
        st.plotly_chart(fig_bertopic_trends, use_container_width=True)

    # --- Additional BERTopic Visualizations ---
    st.subheader("BERTopic Topic Hierarchy")
    with st.spinner("Generating topic hierarchy visualization..."):
        try:
            fig_hierarchy = topic_model.visualize_hierarchy()
            fig_hierarchy.update_layout(
                title=f"{current_domain} - Topic Hierarchy",
                width=800,
                height=600
            )
            st.plotly_chart(fig_hierarchy, use_container_width=True)
        except Exception as e:
            st.info("Hierarchy visualization not available for current topic structure")

    st.subheader("BERTopic Topic Similarity")
    with st.spinner("Generating topic similarity visualization..."):
        try:
            fig_similarity = topic_model.visualize_heatmap()
            fig_similarity.update_layout(
                title=f"{current_domain} - Topic Similarity Heatmap",
                width=700,
                height=700
            )
            st.plotly_chart(fig_similarity, use_container_width=True)
        except Exception as e:
            st.info("Similarity heatmap not available for current topic structure")


    st.header("2. Trend Forecasting (Market Share Projection)")
    st.write("Forecasting future research **market share** based on historical LDA topic trends using **Linear Regression**.")
    
    # Calculate percentage market share instead of absolute counts
    topic_year_percentage = df.groupby(['year', 'Dominant_Topic_Name']).size().unstack(fill_value=0)
    topic_year_percentage = topic_year_percentage.div(topic_year_percentage.sum(axis=1), axis=0) * 100
    
    # --- Forecasting Market Share ---
    future_years = [df['year'].max() + 1, df['year'].max() + 2] # Forecast 1 and 2 years ahead
    all_years = list(topic_year_percentage.index)
    predictions = {}
    
    if len(all_years) < 2:
        st.warning("Not enough distinct publication years in the fetched data to perform trend forecasting.")
    else:
        for topic in topic_year_percentage.columns:
            X = np.array(all_years).reshape(-1, 1)
            y = topic_year_percentage[topic].values
            model = LinearRegression()
            model.fit(X, y)
            forecast = model.predict(np.array(future_years).reshape(-1, 1))
            # Ensure predictions are reasonable percentages (0-100)
            predictions[topic] = np.maximum(0, np.minimum(100, forecast)).round(1)

        # Convert predictions to a DataFrame for easy plotting
        forecast_df = pd.DataFrame(predictions, index=future_years)
        
        # Combine actual and forecast dataframes for one comprehensive plot
        actual_df = topic_year_percentage.reset_index().melt(id_vars='year', var_name='Topic', value_name='Market Share %')
        forecast_df_melted = forecast_df.reset_index().melt(id_vars='index', var_name='Topic', value_name='Market Share %')
        forecast_df_melted = forecast_df_melted.rename(columns={'index': 'year'})
        
        actual_df['Type'] = 'Actual'
        forecast_df_melted['Type'] = 'Forecast'
        
        combined_df = pd.concat([actual_df, forecast_df_melted])

        # Plotting Forecast
        fig_forecast = go.Figure()
        
        for topic in topic_year_percentage.columns:
            # Actual Line
            topic_actual = combined_df[(combined_df['Topic'] == topic) & (combined_df['Type'] == 'Actual')]
            fig_forecast.add_trace(go.Scatter(
                x=topic_actual['year'], 
                y=topic_actual['Market Share %'], 
                mode='lines+markers', 
                name=f"{topic} (Actual)"
            ))
            
            # Forecast Line
            topic_forecast = combined_df[(combined_df['Topic'] == topic) & (combined_df['Type'] == 'Forecast')]
            fig_forecast.add_trace(go.Scatter(
                x=topic_forecast['year'], 
                y=topic_forecast['Market Share %'], 
                mode='lines+markers', 
                line=dict(dash='dash'),
                name=f"{topic} (Forecast)"
            ))

        fig_forecast.update_layout(
            title=f'{current_domain} - Research Market Share Forecast ({future_years[0]}-{future_years[1]})',
            xaxis_title='Year',
            yaxis_title='Market Share (%)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        
        st.subheader(f"Forecasted Market Share ({future_years[0]}-{future_years[1]})")
        
        # Calculate percentage point changes
        current_year = df['year'].max()
        current_shares = topic_year_percentage.loc[current_year]
        forecast_table_data = []
        
        for topic, forecast in predictions.items():
            current_share = current_shares.get(topic, 0)
            change_2026 = forecast[0] - current_share
            change_2027 = forecast[1] - current_share
            
            forecast_table_data.append([
                topic, 
                f"{current_share:.1f}%",
                f"{forecast[0]:.1f}%", 
                f"{change_2026:+.1f}pp",
                f"{forecast[1]:.1f}%", 
                f"{change_2027:+.1f}pp"
            ])
            
        forecast_table = pd.DataFrame(
            forecast_table_data, 
            columns=['Topic', 'Current Share', 
                    f'{future_years[0]} Forecast', f'{future_years[0]} Change', 
                    f'{future_years[1]} Forecast', f'{future_years[1]} Change']
        )
        st.dataframe(forecast_table.set_index('Topic'))

        # --- AI-Powered Insights Section ---
        st.header("3. 🤖 AI-Powered Strategic Insights")
        
        if enable_ai_insights and groq_client:
            with st.spinner("Generating AI-powered insights..."):
                # Prepare data for AI
                top_topics = forecast_df.iloc[-1].sort_values(ascending=False).head(3).index.tolist()
                
                forecast_data = {}
                for topic in top_topics:
                    current_share = current_shares.get(topic, 0)
                    forecast_share = forecast_df.iloc[-1][topic]
                    growth = forecast_share - current_share
                    forecast_data[topic] = {
                        'current': current_share,
                        'forecast': forecast_share,
                        'growth': growth
                    }
                
                future_scope_data = {}
                for topic in top_topics:
                    topic_papers = df[df['Dominant_Topic_Name'] == topic]
                    papers_with_scope = topic_papers[topic_papers['future_scope'].notna()]
                    future_scope_data[topic] = {
                        'count': len(papers_with_scope),
                        'samples': papers_with_scope['future_scope'].head(3).tolist() if len(papers_with_scope) > 0 else []
                    }
                
                # Generate main AI insights
                ai_insights = generate_ai_insights(
                    domain=current_domain,
                    top_topics=top_topics,
                    forecast_data=forecast_data,
                    future_scope_data=future_scope_data,
                    trend_analysis=f"Based on {len(df)} papers from 2020-2025"
                )
                
                st.markdown("### 🎯 AI-Generated Strategic Recommendations")
                st.markdown(ai_insights)
                
                # Generate topic-specific recommendations
                st.markdown("### 🔍 Topic-Specific AI Recommendations")
                for i, topic in enumerate(top_topics):
                    with st.expander(f"AI Insights for: {topic}", expanded=i==0):
                        topic_papers = df[df['Dominant_Topic_Name'] == topic]
                        future_scopes = topic_papers[topic_papers['future_scope'].notna()]['future_scope'].tolist()
                        current_share = current_shares.get(topic, 0)
                        forecast_share = forecast_df.iloc[-1][topic]
                        growth = forecast_share - current_share
                        
                        topic_recommendations = generate_topic_specific_recommendations(
                            domain=current_domain,
                            topic_name=topic,
                            future_scopes=future_scopes,
                            forecast_growth=f"{growth:+.1f} percentage points"
                        )
                        
                        st.markdown(topic_recommendations)
        else:
            st.info("Enable AI insights in the sidebar and configure Groq API key for AI-powered recommendations")
        
        # --- Traditional Analysis (Fallback) ---
        st.header("4. 📊 Traditional Analysis Summary")
        
        top_topics = forecast_df.iloc[-1].sort_values(ascending=False).head(3).index.tolist()
        
        if top_topics:
            st.markdown(f"**Top Predicted High-Growth Areas for {future_years[0]}-{future_years[1]}:**")
            
            for i, topic in enumerate(top_topics):
                # Get papers from this topic that have future scope
                topic_papers = df[df['Dominant_Topic_Name'] == topic]
                papers_with_future_scope = topic_papers[topic_papers['future_scope'].notna()]
                current_share = current_shares.get(topic, 0)
                forecast_share = forecast_df.iloc[-1][topic]
                growth = forecast_share - current_share
                
                st.markdown(f"**{i+1}. {topic}**")
                st.markdown(f"   - Projected to grow from **{current_share:.1f}%** to **{forecast_share:.1f}%** market share by {future_years[1]} ({growth:+.1f} percentage points)")
                st.markdown(f"   - {len(papers_with_future_scope)} papers contain future scope/conclusions")
                
                # Show key future directions from this topic
                if len(papers_with_future_scope) > 0:
                    st.markdown("   - **Key future directions from papers:**")
                    future_scopes = papers_with_future_scope['future_scope'].head(3).tolist()
                    for j, scope in enumerate(future_scopes):
                        # Extract first sentence or first 150 characters
                        first_sentence = scope.split('.')[0] + '.' if '.' in scope else scope[:150] + '...'
                        st.markdown(f"     * {first_sentence}")
                st.markdown("")  # Add spacing between topics
            
            # --- Enhanced Strategic Recommendations ---
            st.markdown("""
            ---
            **Enhanced Strategic Recommendations:**

            * **Focus Allocation:** Prioritize R&D resources towards the top-ranked topics as they show the strongest upward trend in the academic community.
            
            * **Future Scope Alignment:** Align your research directions with the future work identified in existing papers to build upon current knowledge gaps.
            
            * **Talent Acquisition:** Recruit specialists in the predicted high-growth areas to build competitive advantage.
            
            * **Collaboration Opportunities:** Identify potential academic collaborations by examining papers with well-defined future research directions.
            
            * **Monitoring:** Continuously monitor the BERTopic visualization for emerging clusters, as these often represent truly novel, cutting-edge research.
            """)
            
            # --- Future Scope Analysis Summary ---
            st.subheader("📊 Future Scope Analysis Summary")
            future_scope_by_topic = df.groupby('Dominant_Topic_Name')['future_scope'].apply(
                lambda x: x.notna().sum()
            ).reset_index()
            future_scope_by_topic.columns = ['Topic', 'Papers with Future Scope']
            future_scope_by_topic['Total Papers'] = future_scope_by_topic['Topic'].map(df['Dominant_Topic_Name'].value_counts())
            future_scope_by_topic['Future Scope %'] = (future_scope_by_topic['Papers with Future Scope'] / future_scope_by_topic['Total Papers'] * 100).round(1)
            
            # Highlight topics with highest future scope percentage
            st.dataframe(future_scope_by_topic.sort_values('Future Scope %', ascending=False).style.format({'Future Scope %': '{:.1f}%'}))
            
        else:
            st.warning("Forecasting failed or resulted in zero papers. Cannot provide a suggestion.")