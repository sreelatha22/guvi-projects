# Layer 1: Data Engineering (The Foundation)
# Write a PostgreSQL query that transforms 6 raw CSVs into one "Master Analytical View."

# Step A (Squash), Step B (Join), Step C (Calculate): Create the ROAS metric: Revenue / Spend.

import pandas as pd
import glob
import os   
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from psycopg2 import sql
from psycopg2.extras import execute_values
from dotenv import load_dotenv
load_dotenv()
# Database connection parameters from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
def create_db_engine():
    """Create a SQLAlchemy engine for PostgreSQL."""
    try:
        engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        return engine
    except SQLAlchemyError as e:
        print(f"Error creating database engine: {e}")
        return None
def load_csv_files_to_db(engine, csv_folder):
    """Load CSV files from a folder into PostgreSQL database."""
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            table_name = os.path.splitext(os.path.basename(file))[0]
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Loaded {file} into {table_name} table.")
        except Exception as e:
            print(f"Error loading {file}: {e}")
def main():
    # Create database engine
    engine = create_db_engine()
    if engine is None:
        return
    # Load CSV files into the database
    csv_folder = '/Users/sreelatha/code/guvi-projects/Final project - Blinkit/Blinkit - blinkit_customer_feedback.csv'  
    load_csv_files_to_db(engine, csv_folder)
    # Here you would add the SQL transformations to create the Master Analytical View
    # and calculate ROAS metric.
if __name__ == "__main__":
    main()

# Layer 2: Data Analysis (The Insights)
# Using the Master Analytical View from Layer 1, analyze ROAS across different dimensions.
# Identify trends, patterns, and anomalies in ROAS over time, by campaign, and by channel.
# Generate visualizations (e.g., line charts, bar charts) to illustrate your findings   
import matplotlib.pyplot as plt
def analyze_roas(engine):
    """Analyze ROAS from the Master Analytical View."""
    query = "SELECT * FROM master_analytical_view;"  # Replace with actual view name
    try:
        df = pd.read_sql(query, engine)
        # Example analysis: ROAS over time
        df['date'] = pd.to_datetime(df['date'])
        roas_over_time = df.groupby('date')['roas'].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(roas_over_time.index, roas_over_time.values)
        plt.title('ROAS Over Time')
        plt.xlabel('Date')
        plt.ylabel('ROAS')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error analyzing ROAS: {e}")
# Call analyze_roas(engine) in main() after loading data and creating the Master Analytical View
    analyze_roas(engine)

# Layer 3: Data Product (The Application)
# Develop a simple web application using Flask or Django that allows users to interactively explore ROAS data.
# Features should include filtering by date range, campaign, and channel, and displaying visualizations
from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/analyze', methods=['POST'])
def analyze():
    date_range = request.form.get('date_range')
    campaign = request.form.get('campaign')
    channel = request.form.get('channel')
    # Perform analysis based on user input and return results
    # This is a placeholder; actual implementation would query the database
    results = f"Analyzing ROAS for {campaign} on {channel} from {date_range}"
    return render_template('results.html', results=results)
if __name__ == '__main__':
    app.run(debug=True)
# Call analyze_roas(engine) in main() after loading data and creating the Master Analytical View
analyze_roas(engine)

# Layer 2: The Analytics Dashboard (The "Rear-View Mirror")
# Students will build an interactive UI using Python.
#Dual-Axis Charts: On the X-Axis is Time (Days). The Left Y-Axis is
#Revenue (Green Line). The Right Y-Axis is Ad Spend (Red Bars).
#The Insight: If the Red Bar goes UP, but the Green Line stays FLAT, the
#student has visually proven that the marketing campaign is failing.    

import matplotlib.pyplot as plt
def plot_revenue_vs_spend(dates, revenue, ad_spend):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue', color='green')
    ax1.plot(dates, revenue, color='green', label='Revenue')
    ax1.tick_params(axis='y', labelcolor='green')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ad Spend', color='red')
    ax2.bar(dates, ad_spend, color='red', alpha=0.6, label='Ad Spend')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title('Revenue vs Ad Spend Over Time')
    fig.tight_layout()
    plt.show()

# Date from csv file
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
revenue = [1000 + i * 50 for i in range(30)]
ad_spend = [200 + i * 20 for i in range(30)]
plot_revenue_vs_spend(dates, revenue, ad_spend) 
# The Insight: If the Red Bar goes UP, but the Green Line stays FLAT, the
# student has visually proven that the marketing campaign is failing.

# Layer 3: The Data Product (The "Windshield")
#Train the model in Python (scikit-learn).
#Save it as a .pkl file.
#Build a "Risk Calculator" in the App: The manager types "Indiranagar, 6PM",
# and the app says: "⚠ High Risk of Delay (85%).

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Sample data
data = {
    'location': ['Indiranagar', 'MG Road', 'Koramangala', 'Whitefield'] * 25,
    'time_of_day': ['6PM', '12PM', '9AM', '3PM'] * 25,
    'delay': [1,0, 1, 0] * 25
}
df = pd.DataFrame(data)
# Preprocess data
df = pd.get_dummies(df, columns=['location', 'time_of_day'])
X = df.drop('delay', axis=1)
y = df['delay']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Evaluate model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
# Save model
with open('delay_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# Load model and make prediction
def predict_delay_risk(location, time_of_day):
    with open('delay_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    input_data = pd.DataFrame({
        'location_Indiranagar': [1 if location == 'Indiranagar' else 0],
        'location_MG Road': [1 if location == 'MG Road' else 0],
        'location_Koramangala': [1 if location == 'Koramangala' else 0],
        'location_Whitefield': [1 if location == 'Whitefield' else 0],
        'time_of_day_6PM': [1 if time_of_day == '6PM' else 0],
        'time_of_day_12PM': [1 if time_of_day == '12PM' else 0],
        'time_of_day_9AM': [1 if time_of_day == '9AM' else 0],
        'time_of_day_3PM': [1 if time_of_day == '3PM' else 0],
    })
    risk = model.predict_proba(input_data)[:, 1][0]
    return risk
# Example prediction
risk = predict_delay_risk('Indiranagar', '6PM')
print(f"Predicted Delay Risk: {risk*100:.2f}%")
# Build a "Risk Calculator" in the App: The manager types "Indiranagar, 6PM",
# and the app says: "⚠ High Risk of Delay (85%).
from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    time_of_day = request.form.get('time_of_day')
    risk = predict_delay_risk(location, time_of_day)
    return render_template('result.html', risk=risk*100)
if __name__ == '__main__':
    app.run(debug=True)
# Build a "Risk Calculator" in the App: The manager types "Indiranagar, 6PM",
# and the app says: "⚠ High Risk of Delay (85%).   
risk = predict_delay_risk('Indiranagar', '6PM')
print(f"Predicted Delay Risk: {risk*100:.2f}%")

#Layer 4: Generative AI & RAG (The "Brain")
#Retrieval (Search): The student converts all feedback text into math vectors (Embeddings).
# Generation (Answer): It sends those specific comments to an LLM (like GPT-3.5 or Llama) with a prompt
#Use langchain or openai API or Groq API.
#Create a Chat Interface in Streamlit.

import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("feedback_vector_store", embeddings)
# Initialize LLM
llm = OpenAI(temperature=0)
# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())
# Streamlit app 
st.title("Customer Feedback Chatbot")
user_question = st.text_input("Ask a question about customer feedback:")
if user_question:
    answer = qa_chain.run(user_question)
    st.write("Answer:", answer)
# Create a Chat Interface in Streamlit.
st.title("Customer Feedback Chatbot")
user_question = st.text_input("Ask a question about customer feedback:")
if user_question:
    answer = qa_chain.run(user_question)
    st.write("Answer:", answer) 