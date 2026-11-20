import os
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import nltk

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in STOP_WORDS and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf = pickle.load(
    open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb')
)
model = pickle.load(
    open(os.path.join(BASE_DIR, 'model.pkl'), 'rb')
)

st.title("Spam Detection")
input_sms = st.text_input("Enter the message")
# st.write("Message:", input_sms)


if st.button("Predict"):
    # 1. preprocess
    transform_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam⚠️🚨")
    else:
        st.header("Not Spam")


