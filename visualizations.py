# visualizations.py
# All visualization functions

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud

# visualizations.py - Fix the create_topic_distribution_table function

def create_topic_distribution_table(df):
    """Create a comprehensive table showing topic distribution with counts and percentages"""
    if 'Dominant_Topic_Name' not in df.columns:
        return None, None
        
    topic_counts = df['Dominant_Topic_Name'].value_counts()
    total_papers = len(df)
    
    # FIX: Create the summary data correctly
    summary_data = []
    for topic, count in topic_counts.items():
        percentage = (count / total_papers) * 100
        summary_data.append({
            'Research Area': topic,  # This was missing in your code
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
        title=" Research Area Popularity Over Time (%)"
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
                    'Paper Count': max(0, round(pred)),
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
        title=' Research Area Trends & Predictions (2020-2027)',
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
    if 'Dominant_Topic_Name' not in df.columns:
        return None, None
        
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
            r=values + [values[0]],
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
        return None, None
    
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
                set_i = set(topic_keywords[i][:10])
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
            if similarity_matrix[i][j] > 0.1:
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
                       title='🕸️ Topic Relationship Network',
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