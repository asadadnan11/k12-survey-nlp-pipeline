#!/usr/bin/env python3
"""
Launch script for K-12 Survey NLP Dashboard
Interactive analytics dashboard for educational survey insights
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        import sklearn
        import nltk
        import vaderSentiment
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    if not check_dependencies():
        return
    
    print("🚀 Launching K-12 Survey NLP Dashboard...")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("🔍 Features available:")
    print("   • Interactive data filtering")
    print("   • Real-time sentiment analysis")
    print("   • Keyword extraction & TF-IDF scoring")
    print("   • Product impact analysis")
    print("   • Strategic recommendations")
    print("\n💡 Use Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard() 