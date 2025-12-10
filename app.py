import pickle
from flask import Flask, render_template, request
import re
from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__)

# Load pickled files & data
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Stopwords
STOP_WORDS = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","yourselves","he","him","his","himself","she",
    "her","hers","herself","it","its","itself","they","them","their",
    "theirs","themselves","what","which","who","whom","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing","a","an","the",
    "and","but","if","or","because","as","until","while","of","at","by",
    "for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out",
    "on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few",
    "more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don",
    "should","now","d","ll","m","o","re","ve","y","ain","aren","couldn",
    "didn","doesn","hadn","hasn","haven","isn","ma","mightn","mustn",
    "needn","shan","shouldn","wasn","weren","won","wouldn",
    # Custom stopwords
    "fig","figure","image","sample","using","show","result","large",
    "also","one","two","three","four","five","seven","eight","nine"
])

def preprocess_text(txt):
    # Lowercase
    txt = txt.lower()
    # Remove HTML tags
    txt = re.sub(r"<.*?>", " ", txt)
    # Remove special characters and digits
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    # Simple tokenization using split
    tokens = txt.split()
    # Remove stopwords and short words
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) >= 3]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    results = {feature_names[idx]: round(score, 3) for idx, score in sorted_items}
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    document = request.files['file']
    if not document or document.filename == '':
        return render_template('index.html', error='No document selected')

    text = document.read().decode('utf-8', errors='ignore')
    preprocessed_text = preprocess_text(text)
    tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, topn=20)
    return render_template('keywords.html', keywords=keywords)

@app.route('/search_keywords', methods=['POST'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
        keywords = [k for k in feature_names if search_query.lower() in k.lower()][:20]
        return render_template('keywordslist.html', keywords=keywords)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
