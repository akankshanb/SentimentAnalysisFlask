from flask import Flask,render_template,url_for,request
import pickle
import re 
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from service import model

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/analyze',methods=['POST'])
def analyze():
    cv, NB_model = model.train()
    if request.method == 'POST':
        message = request.form['text']
        data = message
        data = preprocess(data)
        vect = cv.transform([data]).toarray()
        predicted_class = NB_model.predict(vect)
        print(predicted_class)
    return render_template('result.html',prediction = predicted_class)

def preprocess(X_test):
    review = re.sub('[^a-zA-Z]', ' ' ,  X_test);
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

if __name__ == "__main__":
    app.run(host='0.0.0.0')
    
