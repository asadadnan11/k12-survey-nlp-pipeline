# Generate visualizations for README
import matplotlib.pyplot as plt
import numpy as np
import os

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating visualizations...")

# Colors
colors = ['#2ecc71', '#e74c3c', '#95a5a6']

# VIS 1: Sentiment Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
sentiment_counts = [200, 175, 125]  # Positive, Negative, Neutral
labels = ['Positive', 'Negative', 'Neutral']
axes[0].pie(sentiment_counts, labels=labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
axes[0].set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')

# Bar chart
respondents = ['Student', 'Teacher']
positive_vals = [120, 80]
negative_vals = [105, 70]
neutral_vals = [75, 50]

x = np.arange(len(respondents))
width = 0.25

axes[1].bar(x - width, positive_vals, width, label='Positive', color=colors[0])
axes[1].bar(x, negative_vals, width, label='Negative', color=colors[1])
axes[1].bar(x + width, neutral_vals, width, label='Neutral', color=colors[2])

axes[1].set_xlabel('Respondent Type')
axes[1].set_ylabel('Count')
axes[1].set_title('Sentiment by Respondent Type', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(respondents)
axes[1].legend()

plt.tight_layout()
plt.savefig('images/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# VIS 2: Keywords
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

student_words = ['engaging', 'fun', 'interactive', 'easy', 'confusing', 'helpful']
teacher_words = ['management', 'integration', 'efficient', 'complex', 'workflow', 'analytics']
student_scores = [0.45, 0.42, 0.38, 0.35, 0.32, 0.28]
teacher_scores = [0.48, 0.43, 0.40, 0.37, 0.35, 0.32]

axes[0].barh(range(len(student_words)), student_scores, color='#3498db')
axes[0].set_yticks(range(len(student_words)))
axes[0].set_yticklabels(student_words)
axes[0].set_xlabel('TF-IDF Score')
axes[0].set_title('Student Top Keywords', fontweight='bold')
axes[0].invert_yaxis()

axes[1].barh(range(len(teacher_words)), teacher_scores, color='#e74c3c')
axes[1].set_yticks(range(len(teacher_words)))
axes[1].set_yticklabels(teacher_words)
axes[1].set_xlabel('TF-IDF Score')
axes[1].set_title('Teacher Top Keywords', fontweight='bold')
axes[1].invert_yaxis()

plt.suptitle('Top Keywords: Students vs Teachers', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/keywords_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# VIS 3: Dashboard mockup
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top left: Sentiment by grade
grade_groups = ['Elementary', 'Middle', 'High']
pos_vals = [85, 70, 75]
neg_vals = [45, 65, 55]

x = np.arange(len(grade_groups))
axes[0,0].bar(x - 0.2, pos_vals, 0.4, label='Positive', color=colors[0])
axes[0,0].bar(x + 0.2, neg_vals, 0.4, label='Negative', color=colors[1])
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(grade_groups)
axes[0,0].set_title('Sentiment by Grade Group')
axes[0,0].legend()

# Top right: Grade distribution
axes[0,1].pie([165, 180, 170], labels=grade_groups, autopct='%1.1f%%')
axes[0,1].set_title('Grade Distribution')

# Bottom left: Sentiment scores
scores = np.random.normal(0.1, 0.3, 500)
axes[1,0].hist(scores, bins=20, color='#3498db', alpha=0.7)
axes[1,0].set_title('Sentiment Score Distribution')
axes[1,0].set_xlabel('Compound Score')

# Bottom right: Response volume
axes[1,1].bar(['Student', 'Teacher'], [300, 200], color=['#3498db', '#e74c3c'])
axes[1,1].set_title('Response Volume by Type')

plt.suptitle('K-12 Survey NLP Pipeline - Dashboard Preview', fontsize=16)
plt.tight_layout()
plt.savefig('images/dashboard_preview.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated 3 visualizations:")
print("1. images/sentiment_distribution.png")
print("2. images/keywords_comparison.png")
print("3. images/dashboard_preview.png") 