# data_fetcher.py
# arXiv data fetching functionality

import streamlit as st
import pandas as pd
import arxiv
import random
import time
import re
from datetime import datetime
from text_processor import extract_future_scope
from config import START_YEAR, END_YEAR

@st.cache_data(ttl=3600)
def fetch_historical_papers(selected_domain, arxiv_category, keywords):
    """Fetches papers with AI-enhanced keyword selection"""
    client = arxiv.Client()
    
    all_papers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_years = END_YEAR - START_YEAR + 1
    
    for year_idx, year in enumerate(range(START_YEAR, END_YEAR + 1)):
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

            st.success(f" Fetched {papers_count} {selected_domain} papers from {year}")
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
        
        # FIX: Use proper string formatting with st.write instead of f-string with emoji
        st.write(f"**{selected_domain} papers per year:**")
        for year, count in year_counts.items():
            st.write(f"  {year}: {count} papers")
        
        future_scope_found = df['future_scope'].notna().sum()
        percentage = (future_scope_found / len(df)) * 100
        
        # FIX: Use st.write with proper formatting
        st.write(f"**Future scope/conclusion found in {future_scope_found} out of {len(df)} papers ({percentage:.1f}%)**")
        
        return df
    else:
        st.error(f" No {selected_domain} papers were fetched")
        return pd.DataFrame()