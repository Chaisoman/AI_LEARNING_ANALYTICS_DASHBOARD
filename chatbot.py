import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

class LearningChatbot:
    def __init__(self):
        self.model = joblib.load('recommendation_model.pkl')
        self.recommendation_map = joblib.load('recommendation_map.pkl')
        self.data = pd.read_csv('personalized_learning_dataset.csv')

    def preprocess_input(self, student_data):
        df = student_data.copy()
        le = LabelEncoder()
        for col in ['Learning_Style', 'Dropout_Likelihood', 'Education_Level', 'Engagement_Level', 'Gender']:
            if col in df:
                df[col] = le.fit_transform(df[col])
        # Drop non-numeric columns, ensure no 'Recommendation' column
        return df.drop(['Student_ID', 'Course_Name', 'Recommendation'], axis=1, errors='ignore')

    def get_recommendation(self, student_id, for_student_tab=False):
        student = self.data[self.data['Student_ID'] == student_id]
        if student.empty:
            return "Student not found."
        try:
            X = self.preprocess_input(student)
            pred = self.model.predict(X)[0]
            base_recommendation = self.recommendation_map.get(pred, "General review sessions recommended.")
            # Add specific tips based on student data
            student = student.iloc[0]
            additional_tips = []
            if student['Time_Spent_on_Videos'] < 250:
                additional_tips.append("Watch more videos online to reinforce learning concepts.")
            if student['Forum_Participation'] < 5:
                additional_tips.append("Link up with peers in forums or study groups to boost engagement.")
            if student['Assignment_Completion_Rate'] < 70:
                additional_tips.append("Prioritize completing assignments to improve understanding.")
            tips_text = " Additional tips: " + " ".join(additional_tips) if additional_tips else ""
            
            if for_student_tab:
                # Add performance explanation
                performance = (f"Your performance summary: Feedback Score ({student['Feedback_Score']}/5) reflects course satisfaction, "
                              f"Final Exam Score ({student['Final_Exam_Score']}/100) shows mastery, "
                              f"Quiz Scores ({student['Quiz_Scores']}/100) indicate short-term retention, "
                              f"Assignment Completion ({student['Assignment_Completion_Rate']}%) shows effort, "
                              f"and Forum Participation ({student['Forum_Participation']}) reflects engagement.")
                return f"{base_recommendation}{tips_text}\n\n{performance}"
            return f"{base_recommendation}{tips_text}"
        except Exception as e:
            return f"Error generating recommendation: {e}"

    def respond(self, student_id, query):
        if not student_id:
            return "Please provide a Student ID."
        student = self.data[self.data['Student_ID'] == student_id]
        if student.empty:
            return "Student not found."
        
        recommendation = self.get_recommendation(student_id)
        if "tips" in query.lower() or "recommend" in query.lower():
            return f"Recommendation: {recommendation}"
        elif "performance" in query.lower():
            student = student.iloc[0]
            return (f"Performance for {student_id}: "
                    f"Feedback Score: {student['Feedback_Score']}, "
                    f"Final Exam Score: {student['Final_Exam_Score']}, "
                    f"Quiz Scores: {student['Quiz_Scores']}, "
                    f"Assignment Completion: {student['Assignment_Completion_Rate']}%")
        elif "dropout" in query.lower():
            return f"Dropout Likelihood for {student_id}: {student.iloc[0]['Dropout_Likelihood']}"
        else:
            return f"Recommendation: {recommendation}\nAsk about tips, performance, or dropout risk for specific advice."
