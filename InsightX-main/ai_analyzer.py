# ai_analyzer.py
# AI-powered analysis functions

from groq_client import generate_with_groq

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

def parse_domain_suggestions(suggestions):
    """Parse AI-generated domain suggestions (simplified implementation)"""
    domains = {}
    lines = suggestions.split('\n')
    for line in lines:
        if 'cs.' in line.lower():
            parts = line.split('-')
            if len(parts) >= 2:
                domain = parts[0].strip()
                category = parts[1].strip()
                domains[domain] = category
    return domains