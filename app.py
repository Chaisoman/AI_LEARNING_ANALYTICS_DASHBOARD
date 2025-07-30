from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from chatbot import LearningChatbot
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset and chatbot
data = pd.read_csv('personalized_learning_dataset.csv')
chatbot = LearningChatbot()

@app.route('/')
def index():
    return render_template('index.html', students=data.to_dict(orient='records'))

@app.route('/student', methods=['GET', 'POST'])
def student():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        student = data[data['Student_ID'] == student_id].to_dict(orient='records')
        try:
            recommendation = chatbot.get_recommendation(student_id, for_student_tab=True)
            chat_response = chatbot.respond(student_id, request.form.get('prompt', '')) if request.form.get('prompt') else None
            if student:
                perf_data = {
                    'Metrics': ['Feedback Score', 'Final Exam Score', 'Quiz Scores', 'Assignment Completion Rate', 'Forum Participation'],
                    'Values': [
                        student[0]['Feedback_Score'],
                        student[0]['Final_Exam_Score'],
                        student[0]['Quiz_Scores'],
                        student[0]['Assignment_Completion_Rate'],
                        student[0]['Forum_Participation']
                    ]
                }
                radar_fig = px.line_polar(
                    pd.DataFrame(perf_data),
                    r='Values',
                    theta='Metrics',
                    line_close=True,
                    title=f"Performance for {student_id}"
                )
                radar_html = pio.to_html(radar_fig, full_html=False)
            else:
                radar_html = None
        except Exception as e:
            logging.error(f"Error: {e}")
            chat_response = f"Error processing request: {e}"
            recommendation = f"Error generating recommendation: {e}"
            radar_html = None
        return render_template('student.html', 
                             student=student[0] if student else None,
                             chat_response=chat_response,
                             student_id=student_id,
                             recommendation=recommendation,
                             radar_html=radar_html)
    return render_template('student.html', student=None, chat_response=None, recommendation=None, radar_html=None)

@app.route('/management', methods=['GET', 'POST'])
def management():
    student_id = request.form.get('student_id') if request.method == 'POST' else None
    filtered_data = data[data['Student_ID'] == student_id] if student_id else data
    # Visuals
    bar_fig = px.bar(filtered_data, x='Course_Name', y='Feedback_Score', title='Feedback Score by Course')
    bar_html = pio.to_html(bar_fig, full_html=False)
    pie_fig = px.pie(filtered_data, names='Learning_Style', title='Learning Style Distribution')
    pie_html = pio.to_html(pie_fig, full_html=False)
    # Dropout Likelihood by Course and Education Level
    dropout_fig = px.bar(
        filtered_data, 
        x='Course_Name', 
        y='Feedback_Score', 
        color='Dropout_Likelihood', 
        facet_col='Education_Level',
        title='Dropout Likelihood by Course and Education Level',
        barmode='group'
    )
    dropout_html = pio.to_html(dropout_fig, full_html=False)
    # Demographics: Gender, Age, Course
    demo_fig = px.histogram(
        filtered_data,
        x='Course_Name',
        color='Gender',
        facet_col='Age',
        title='Student Demographics by Course',
        barmode='group'
    )
    demo_html = pio.to_html(demo_fig, full_html=False)
    # Performance Metrics: Feedback, Assignment, Forum, Exam
    if student_id and not filtered_data.empty:
        perf_data = {
            'Metrics': ['Feedback Score', 'Assignment Completion', 'Forum Participation', 'Final Exam Score'],
            'Values': [
                filtered_data['Feedback_Score'].iloc[0],
                filtered_data['Assignment_Completion_Rate'].iloc[0],
                filtered_data['Forum_Participation'].iloc[0],
                filtered_data['Final_Exam_Score'].iloc[0]
            ]
        }
        perf_fig = px.line_polar(
            pd.DataFrame(perf_data),
            r='Values',
            theta='Metrics',
            line_close=True,
            title='Performance Metrics Comparison'
        )
    else:
        # Aggregated performance for multiple students
        perf_data = {
            'Metrics': ['Feedback Score', 'Assignment Completion', 'Forum Participation', 'Final Exam Score'],
            'Values': [
                filtered_data['Feedback_Score'].mean(),
                filtered_data['Assignment_Completion_Rate'].mean(),
                filtered_data['Forum_Participation'].mean(),
                filtered_data['Final_Exam_Score'].mean()
            ]
        }
        perf_fig = px.line_polar(
            pd.DataFrame(perf_data),
            r='Values',
            theta='Metrics',
            line_close=True,
            title='Performance Metrics Comparison (Aggregated)'
        )
    perf_html = pio.to_html(perf_fig, full_html=False)
    
    if request.method == 'POST':
        try:
            recommendation = chatbot.get_recommendation(student_id) if student_id else "Review aggregated data for general recommendations."
            chat_response = chatbot.respond(student_id, request.form.get('prompt', '')) if request.form.get('prompt') else None
        except Exception as e:
            logging.error(f"Error: {e}")
            chat_response = f"Error processing request: {e}"
            recommendation = f"Error generating recommendation: {e}"
        return render_template('management.html', 
                             chat_response=chat_response,
                             bar_html=bar_html,
                             pie_html=pie_html,
                             dropout_html=dropout_html,
                             demo_html=demo_html,
                             perf_html=perf_html,
                             recommendation=recommendation,
                             student_id=student_id)
    return render_template('management.html', 
                         chat_response=None,
                         bar_html=bar_html,
                         pie_html=pie_html,
                         dropout_html=dropout_html,
                         demo_html=demo_html,
                         perf_html=perf_html,
                         recommendation=None,
                         student_id=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
