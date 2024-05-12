from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('C:/Users/maury/Desktop/Code/project/Job matching model/flask_app/UpdatedResumeDataSet.csv')

df['Merged'] = df['Category'] + ': ' + df['Resume']

# Drop the "Category" column
df.drop(columns=['Category'], inplace=True)

# Display the DataFrame with the merged column and without the "Category" column
df.drop(columns=['Resume'], inplace=True)

# Preprocess text function
def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    text = ' '.join(filtered_tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(lemmatized_tokens)

    return text

# Calculate similarity function
def calculate_similarity(job_description, df, threshold=0.1, max_features=None):
    # Preprocess job description
    job_description = preprocess_text(job_description)

    # TF-IDF Vectorization with maximum features
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df['Merged'].tolist() + [job_description])

    # Cosine Similarity Calculation
    similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

    # Create DataFrame with Similarity Scores
    result_df = df.copy()
    result_df['Similarity'] = similarity_scores

    # Add target column based on similarity threshold
    result_df['Target'] = (result_df['Similarity'] >= threshold).astype(int)

    return result_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    job_description = request.form['job_description']
    result_df_max_features = calculate_similarity(job_description, df, max_features=1000)
    
    # Calculate the count of selected resumes
    selected_count = result_df_max_features['Target'].sum()

    # Pass the selected count to the template
    return render_template('result.html', selected_count=selected_count)

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/redirect-url')
def redirect_url():
    # Perform any necessary operations here
    # Redirect to the desired URL
    return redirect("https://docs.google.com/spreadsheets/d/19E857TiOliXkFhO421lnQxP-z--rOfzCGklHL7gJjAY/edit?usp=sharing")  # Replace "https://example.com" with the desired URL


if __name__ == '__main__':
    app.run(debug=True)
