# ============================================================================
# VISUALIZATION UTILITIES
# Functions for creating charts and visualizations
# ============================================================================

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import base64
from io import BytesIO


# Color schemes
SENTIMENT_COLORS = {
    'positive': '#2ecc71',
    'neutral': '#95a5a6', 
    'negative': '#e74c3c'
}


def plot_sentiment_distribution(df, title="Sentiment Distribution"):
    """
    Create bar chart of sentiment distribution
    """
    sentiment_counts = df['sentiment_category'].value_counts()
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Count'},
        title=title,
        color=sentiment_counts.index,
        color_discrete_map=SENTIMENT_COLORS
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Sentiment Category",
        yaxis_title="Number of Texts"
    )
    
    return fig


def plot_sentiment_pie(df, title="Sentiment Proportion"):
    """
    Create pie chart of sentiment distribution
    """
    sentiment_counts = df['sentiment_category'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title=title,
        color=sentiment_counts.index,
        color_discrete_map=SENTIMENT_COLORS
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    
    return fig


def plot_confidence_distribution(df, title="Confidence Level Distribution"):
    """
    Create bar chart of confidence levels
    """
    confidence_order = ['low', 'medium', 'high']
    confidence_counts = df['confidence_level'].value_counts().reindex(confidence_order, fill_value=0)
    
    fig = px.bar(
        x=confidence_counts.index,
        y=confidence_counts.values,
        labels={'x': 'Confidence Level', 'y': 'Count'},
        title=title,
        color=confidence_counts.index,
        color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71']
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Confidence Level",
        yaxis_title="Number of Texts"
    )
    
    return fig


def plot_sentiment_by_confidence(df, title="Sentiment by Confidence Level"):
    """
    Create grouped bar chart showing sentiment distribution per confidence level
    """
    # Create crosstab
    crosstab = pd.crosstab(df['confidence_level'], df['sentiment_category'])
    
    # Reorder confidence levels
    confidence_order = ['low', 'medium', 'high']
    if all(level in crosstab.index for level in confidence_order):
        crosstab = crosstab.reindex(confidence_order)
    
    fig = go.Figure()
    
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment in crosstab.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=crosstab.index,
                y=crosstab[sentiment],
                marker_color=SENTIMENT_COLORS[sentiment]
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence Level",
        yaxis_title="Count",
        barmode='group',
        template="plotly_white",
        legend_title="Sentiment"
    )
    
    return fig


def plot_text_length_distribution(df, text_column, title="Text Length Distribution"):
    """
    Create histogram of text lengths
    """
    text_lengths = df[text_column].str.len()
    
    fig = px.histogram(
        x=text_lengths,
        nbins=50,
        labels={'x': 'Text Length (characters)', 'y': 'Frequency'},
        title=title,
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def plot_word_count_distribution(df, text_column, title="Word Count Distribution"):
    """
    Create histogram of word counts
    """
    word_counts = df[text_column].str.split().str.len()
    
    fig = px.histogram(
        x=word_counts,
        nbins=30,
        labels={'x': 'Word Count', 'y': 'Frequency'},
        title=title,
        color_discrete_sequence=['#9b59b6']
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def generate_wordcloud(text, colormap='viridis', width=800, height=400):
    """
    Generate word cloud and return as base64 image
    """
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Convert to image
        img = wordcloud.to_image()
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return None


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Create heatmap confusion matrix
    """
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='RdYlGn',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False,
        colorbar=dict(title="Normalized Value")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_white",
        width=600,
        height=500
    )
    
    return fig


def plot_roc_curves(roc_data, class_names, title="ROC Curves"):
    """
    Create ROC curves for multiclass classification
    """
    fig = go.Figure()
    
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    
    for i, class_name in enumerate(class_names):
        if class_name in roc_data:
            data = roc_data[class_name]
            fig.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                mode='lines',
                name=f"{class_name} (AUC = {data['auc']:.3f})",
                line=dict(color=colors[i], width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend=dict(x=0.6, y=0.1),
        width=700,
        height=600
    )
    
    return fig


def plot_feature_importance(importance_dict, sentiment, top_n=15):
    """
    Plot feature importance for a specific sentiment
    """
    if not importance_dict or sentiment not in importance_dict:
        return None
    
    data = importance_dict[sentiment]
    
    # Get top features
    positive_features = data.get('positive', [])[:top_n]
    
    if not positive_features:
        return None
    
    features = [f[0] for f in positive_features]
    values = [f[1] for f in positive_features]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=SENTIMENT_COLORS.get(sentiment, '#3498db')
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Features for {sentiment.capitalize()} Sentiment",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_model_comparison(comparison_df, title="Model Performance Comparison"):
    """
    Create grouped bar chart comparing model metrics
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Macro']
    
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                marker_color=colors[i]
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        template="plotly_white",
        legend_title="Metric",
        yaxis=dict(range=[0, 1])
    )
    
    return fig