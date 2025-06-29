import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Simple test to see what's failing
st.title("🔧 Dashboard Debug Test")

try:
    st.write("✅ Streamlit is working")
    
    # Test basic data generation
    st.subheader("Testing Data Generation")
    np.random.seed(42)
    test_data = pd.DataFrame({
        'x': range(10),
        'y': np.random.randn(10)
    })
    st.write("✅ Data generation works")
    st.dataframe(test_data.head())
    
    # Test plotting
    st.subheader("Testing Plotly")
    fig = px.line(test_data, x='x', y='y', title='Test Chart')
    st.plotly_chart(fig, use_container_width=True)
    st.write("✅ Plotly charts work")
    
    # Test the actual dashboard data function
    st.subheader("Testing Dashboard Data Function")
    
    from dashboard import generate_enhanced_dataset
    df = generate_enhanced_dataset()
    st.write(f"✅ Generated {len(df)} rows of data")
    st.dataframe(df.head())
    
    # Test sentiment analysis
    from dashboard import analyze_sentiment
    df_with_sentiment = analyze_sentiment(df.head(10))  # Just test 10 rows
    st.write("✅ Sentiment analysis works")
    st.dataframe(df_with_sentiment[['response_text', 'sentiment_label', 'compound']].head())
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e) 