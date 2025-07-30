import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv('personalized_learning_dataset.csv')

# Define recommendation mapping
recommendation_map = {
    0: "Focus on visual aids like diagrams and videos to enhance understanding.",
    1: "Engage in hands-on activities and practical exercises to improve retention.",
    2: "Use written summaries and note-taking to reinforce learning.",
    3: "Incorporate group discussions and interactive forums to boost engagement.",
    4: "Schedule regular review sessions to address low performance areas."
}

# Preprocess data
def preprocess_data(data):
    df = data.copy()
    # Encode categorical variables
    le = LabelEncoder()
    df['Learning_Style'] = le.fit_transform(df['Learning_Style'])
    df['Dropout_Likelihood'] = le.fit_transform(df['Dropout_Likelihood'])
    df['Education_Level'] = le.fit_transform(df['Education_Level'])
    df['Engagement_Level'] = le.fit_transform(df['Engagement_Level'])
    df['Gender'] = le.fit_transform(df['Gender'])
    # Drop non-numeric and target columns
    df = df.drop(['Student_ID', 'Course_Name', 'Recommendation'], axis=1, errors='ignore')
    return df

# Generate synthetic target (recommendation labels) based on rules
def create_synthetic_target(row):
    if row['Learning_Style'] == 'Visual' and row['Feedback_Score'] >= 4:
        return 0  # Visual aids
    elif row['Learning_Style'] == 'Kinesthetic':
        return 1  # Hands-on activities
    elif row['Learning_Style'] == 'Reading/Writing' and row['Final_Exam_Score'] < 50:
        return 2  # Written summaries
    elif row['Forum_Participation'] > 5:
        return 3  # Group discussions
    else:
        return 4  # Review sessions

# Prepare data
data['Recommendation'] = data.apply(create_synthetic_target, axis=1)
X = preprocess_data(data)
y = data['Recommendation']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model and recommendation map
joblib.dump(model, 'recommendation_model.pkl')
joblib.dump(recommendation_map, 'recommendation_map.pkl')
print("Model and recommendation map saved.")
