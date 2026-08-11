# main.py
# Main Streamlit application

import streamlit as st
import pandas as pd
from config import DEFAULT_DOMAINS
from groq_client import setup_groq_client
from data_fetcher import fetch_historical_papers
from topic_modeler import perform_topic_modeling
from visualizations import (
    create_topic_distribution_table,
    create_topic_evolution_heatmap,
    create_topic_trends_with_predictions,
    create_future_scope_analysis,
    create_research_maturity_chart,
    create_topic_wordclouds,
    create_topic_network_analysis
)
from ai_analyzer import (
    generate_domain_suggestions,
    generate_domain_keywords,
    analyze_topic_trends,
    generate_research_gaps,
    parse_domain_suggestions
)

# --- Streamlit Setup ---
st.set_page_config(page_title="AI-Powered Tech Trend Analyzer", layout="wide")

st.title(" AI-Powered Tech Trend Analyzer")
st.write("**Fully dynamic** research analysis with AI-generated insights, topic labeling, and strategic recommendations using arXiv data.")

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
                domains = parse_domain_suggestions(suggestions)
            else:
                st.sidebar.warning("Using default domains")
    
    # Generate keywords for domains dynamically
    for domain in domains:
        if domain not in st.session_state.get('domain_keywords', {}):
            keyword_status = st.sidebar.empty()
            keyword_status.info(f"Generating keywords for {domain}...")
            keywords = generate_domain_keywords(domain, domains[domain])
            keyword_status.empty()
            
            if keywords:
                domain_keywords[domain] = [k.strip() for k in keywords.split(',')]
            else:
                domain_keywords[domain] = [domain.lower()]
    
    st.session_state.domain_keywords = domain_keywords
    return domains, domain_keywords

def initialize_session_state():
    """Initialize session state variables"""
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

def render_sidebar():
    """Render the sidebar configuration"""
    st.sidebar.header("🔧 AI-Powered Configuration")
    
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

    # Groq API Status
    st.sidebar.header(" AI Features")
    groq_client = setup_groq_client()
    if groq_client:
        st.sidebar.success(" Groq API connected!")
    else:
        st.sidebar.error(" Groq API not configured")

    enable_ai_features = st.sidebar.checkbox("Enable AI-Powered Features", value=True,
                                            help="Use AI for dynamic topic naming, insights, and recommendations")

    return selected_domain, arxiv_category, keywords, enable_ai_features, groq_client

# main.py - Update the render_analysis_dashboard function

def render_analysis_dashboard(df, current_domain, enable_ai_features, groq_client):
    """Render the main analysis dashboard"""
    
    # FIX: Perform topic modeling FIRST before showing stats
    st.header("1.  AI-Powered Topic Modeling")
    
    df, lda_model, dictionary = perform_topic_modeling(df, current_domain, enable_ai_features, groq_client)
    
    # Store models in session state for later use
    st.session_state.lda_model = lda_model
    st.session_state.dictionary = dictionary
    
    # FIX: Update session state with the modified dataframe
    st.session_state.papers = df

    # NOW show the quick stats overview
    st.subheader(" Quick Overview")
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
        # FIX: This will now work because topic modeling is done
        topics_discovered = df['Dominant_Topic_Name'].nunique() if 'Dominant_Topic_Name' in df.columns else 0
        st.metric("Research Areas Found", topics_discovered)

    # Continue with the rest of the dashboard...
    st.header("2.  Interactive Research Dashboard")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " Overview", 
        " Trends", 
        " Future Research", 
        " Maturity", 
        " Keywords", 
        " Relationships"
    ])
    
    # Render each tab
    render_overview_tab(tab1, df)
    render_trends_tab(tab2, df)
    render_future_research_tab(tab3, df)
    render_maturity_tab(tab4, df)
    render_keywords_tab(tab5, df, lda_model, dictionary)
    render_relationships_tab(tab6, df, lda_model, dictionary)

    # Sample Papers Display
    render_sample_papers(df)
    
    # AI-Powered Analysis Section
    if enable_ai_features and groq_client:
        render_ai_insights(df, current_domain)
    
    # Data Export
    render_data_export(df, current_domain)
    
    return df
def render_overview_tab(tab, df):
    """Render the overview tab"""
    with tab:
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
        
        if dist_df is not None:
            with st.expander(" Detailed Research Area Statistics"):
                st.dataframe(dist_df, use_container_width=True)

def render_trends_tab(tab, df):
    """Render the trends tab"""
    with tab:
        st.subheader("Research Trends & Predictions")
        
        trends_chart, predictions_df = create_topic_trends_with_predictions(df)
        if trends_chart:
            st.plotly_chart(trends_chart, use_container_width=True)
            
            if predictions_df is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    with st.expander(" View Predictions Table"):
                        st.write("**Predicted Paper Counts for 2026-2027:**")
                        pivot_df = predictions_df.pivot(index='Research Area', columns='Year', values='Predicted Papers')
                        st.dataframe(pivot_df, use_container_width=True)
                
                with col2:
                    st.metric("Total Predicted Papers (2026-2027)", 
                             int(predictions_df['Predicted Papers'].sum()))
        else:
            st.info("Trend analysis data not available")

def render_future_research_tab(tab, df):
    """Render the future research tab"""
    with tab:
        st.subheader("Future Research Directions Analysis")
        
        future_chart, future_df = create_future_scope_analysis(df)
        if future_chart:
            st.plotly_chart(future_chart, use_container_width=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.expander(" Future Scope Details"):
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

def render_maturity_tab(tab, df):
    """Render the maturity tab"""
    with tab:
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
                    with st.expander("View All Topics Maturity Scores"):
                        display_df = maturity_df.copy()
                        for col in ['Paper Volume', 'Longevity', 'Future Focus', 'Consistency']:
                            display_df[col] = display_df[col].round(1)
                        st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Research maturity analysis data not available")

def render_keywords_tab(tab, df, lda_model, dictionary):
    """Render the keywords tab"""
    with tab:
        st.subheader("Topic Keyword Visualization")
        
        wordcloud_figs = create_topic_wordclouds(df, lda_model, dictionary)
        
        if wordcloud_figs:
            cols = st.columns(2)
            for i, fig in enumerate(wordcloud_figs):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if 'Dominant_Topic_Name' in df.columns:
                        topic_name = df[df['Dominant_Topic_ID'] == i]['Dominant_Topic_Name'].iloc[0]
                        st.caption(f"**{topic_name}** - Top keywords visualized")
        else:
            st.info("Word cloud data not available")

def render_relationships_tab(tab, df, lda_model, dictionary):
    """Render the relationships tab"""
    with tab:
        st.subheader("Topic Relationship Network")
        
        network_fig, similarity_matrix = create_topic_network_analysis(df, lda_model, dictionary)
        
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
                        sim_df = pd.DataFrame(
                            similarity_matrix,
                            index=[f"Topic {i+1}" for i in range(len(similarity_matrix))],
                            columns=[f"Topic {i+1}" for i in range(len(similarity_matrix))]
                        )
                        st.dataframe(sim_df.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("Network analysis data not available")

def render_sample_papers(df):
    """Render sample papers section"""
    st.header("3.  Sample Research Papers")
    
    if 'Dominant_Topic_Name' in df.columns:
        topic_select = st.selectbox("Select Research Area to View Papers:", df['Dominant_Topic_Name'].unique())
        
        topic_papers = df[df['Dominant_Topic_Name'] == topic_select].head(5)
        
        st.write(f"**Showing 5 sample papers from: {topic_select}**")
        
        for idx, paper in topic_papers.iterrows():
            with st.expander(f" {paper['title']}"):
                st.write(f"**Published:** {paper['year']}")
                st.write(f"**Abstract:** {paper['abstract'][:400]}...")
                if paper['future_scope']:
                    st.write(f"**Future Research Directions:** {paper['future_scope'][:300]}...")
                else:
                    st.write("**Future Research Directions:** Not specified in abstract")
    else:
        st.info("Topic information not available")

def render_ai_insights(df, current_domain):
    """Render AI insights section"""
    st.header("4.  AI Research Insights")
    
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

def render_data_export(df, current_domain):
    """Render data export section"""
    st.header("5.  Export Results")
    
    if st.button(" Download Complete Analysis"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name=f"{current_domain.replace(' ', '_')}_analysis.csv",
            mime="text/csv"
        )

def render_welcome_message():
    """Render welcome message for first-time users"""
    st.info("""
    ##  Welcome to AI-Powered Tech Trend Analyzer!
    
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

def main():
    """Main application function"""
    initialize_session_state()
    
    # Render sidebar and get configuration
    selected_domain, arxiv_category, keywords, enable_ai_features, groq_client = render_sidebar()
    
    # Data Fetching
    if st.sidebar.button(f"🚀 Fetch {selected_domain} Papers (2020-2025)"):
        try:
            from nltk.corpus import stopwords
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

    # Display analysis if papers exist
    if st.session_state.papers is not None and not st.session_state.papers.empty:
        current_domain = st.session_state.current_domain
        st.header(f" {current_domain} Analysis Results")
        df = st.session_state.papers
        
        # Render the main analysis dashboard
        render_analysis_dashboard(df, current_domain, enable_ai_features, groq_client)
    else:
        render_welcome_message()

if __name__ == "__main__":
    main()