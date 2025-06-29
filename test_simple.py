import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("🔧 Testing Dashboard Components Independently")

# Test 1: Data Generation
print("\n1. Testing synthetic data generation...")
try:
    np.random.seed(42)
    
    student_responses = [
        "The platform is exhausting to use for long periods, makes me tired",
        "Interface is overwhelming with too many options and buttons",
        "Interactive lessons help me understand concepts better than textbooks",
        "Love the gamification elements, makes learning fun and engaging"
    ]
    
    teacher_responses = [
        "Students seem disengaged after 20 minutes on the platform",
        "Grading interface is unintuitive and slows down my workflow significantly",
        "Automated grading saves me hours of work each week",
        "Parent communication portal keeps families engaged in learning process"
    ]
    
    # Generate small test dataset
    data = []
    for i in range(10):
        if i < 6:  # 60% students
            response = random.choice(student_responses)
            resp_type = 'Student'
        else:
            response = random.choice(teacher_responses)
            resp_type = 'Teacher'
        
        data.append({
            'response_id': f'{resp_type[:3].upper()}_{i+1:03d}',
            'respondent_type': resp_type,
            'grade_level': random.choice(['Elementary', 'Middle', 'High']),
            'response_text': response,
            'response_date': datetime.now() - timedelta(days=random.randint(1, 30)),
            'session_duration_minutes': random.randint(5, 45)
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} test responses")
    print(df[['respondent_type', 'grade_level', 'response_text']].head(3))
    
except Exception as e:
    print(f"❌ Data generation failed: {e}")

# Test 2: Sentiment Analysis
print("\n2. Testing sentiment analysis...")
try:
    analyzer = SentimentIntensityAnalyzer()
    
    sentiments = []
    for text in df['response_text']:
        scores = analyzer.polarity_scores(text)
        sentiments.append({
            'compound': scores['compound'],
            'sentiment_label': 'Positive' if scores['compound'] > 0.05 
                             else 'Negative' if scores['compound'] < -0.05 
                             else 'Neutral'
        })
    
    sentiment_df = pd.DataFrame(sentiments)
    df_with_sentiment = pd.concat([df, sentiment_df], axis=1)
    
    print("✅ Sentiment analysis completed")
    print(df_with_sentiment[['response_text', 'sentiment_label', 'compound']].head(3))
    
except Exception as e:
    print(f"❌ Sentiment analysis failed: {e}")

# Test 3: Basic Statistics
print("\n3. Testing basic analytics...")
try:
    sentiment_dist = df_with_sentiment['sentiment_label'].value_counts()
    avg_sentiment = df_with_sentiment['compound'].mean()
    
    print("✅ Basic analytics working")
    print(f"Sentiment distribution: {dict(sentiment_dist)}")
    print(f"Average sentiment: {avg_sentiment:.3f}")
    
except Exception as e:
    print(f"❌ Analytics failed: {e}")

print("\n🎯 Component test complete!")
print("If all tests passed, the dashboard should work correctly.") 