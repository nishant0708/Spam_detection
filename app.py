import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')  # Add this line


ps = PorterStemmer()

def transform_text(text):
    ## converting to lowercase letter
    text = text.lower()
    ## separating sentences into words
    text = nltk.word_tokenize(text)

    ## removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the Message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])  # Corrected this line
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
