# Keyword-Extraction-Web-Application
This is a Flask-based web application for extracting keywords from text documents using machine learning techniques such as TF-IDF and Count Vectorizer.

---

## Project Structure

Kewords-Extraction-App-machine-Learning-main/
│
├── .venv/ # Python virtual environment
├── nltk_data/ # NLTK datasets and models
├── templates/ # HTML templates for Flask app
│ ├── index.html # Main page with upload and search form
│ ├── keywords.html # Displays extracted keywords
│ └── keywordslist.html # Displays searched keywords list
├── app.py # Flask application
├── count_vectorizer.pkl # Pickled CountVectorizer object
├── cv.pkl # Possibly another pickled vectorizer
├── feature_names.pkl # Pickled feature names for keywords
├── tfidf_transformer.pkl # Pickled TF-IDF transformer
├── papers.csv # Sample document or dataset (optional)
├── README.md # This file
└── Keyword Extraction with Python.ipynb # Jupyter notebook (optional)


---

## Features

- Upload text documents to extract top keywords using TF-IDF weighting.
- Search within the extracted keywords.
- Clean and preprocess input text including tokenization, lemmatization, and stopword removal.
- Interactive frontend with file upload and keyword search forms.
- Uses NLTK for natural language processing.
- Flask backend serving HTML templates with Jinja2.

---

## Requirements

- Python 3.7+
- Flask
- nltk
- scikit-learn
- pickle (part of Python standard library)

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-repo/keyword-extraction-app.git
cd keyword-extraction-app


2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows

3. Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

4. Run the Flask application
python app.py

Access the app at http://127.0.0.1:5000/


Usage

Navigate to the home page.

Upload a .txt file or drag & drop a document to extract keywords.

View the extracted keywords on the results page.

Use the search box to find keywords from the feature names.


Notes

Make sure your .pkl files (count_vectorizer.pkl, tfidf_transformer.pkl, feature_names.pkl) are trained and present in the project root.

The app expects input files encoded in UTF-8.

The preprocessing pipeline uses standard English stopwords plus custom stopwords.



Create and activate virtual environment
