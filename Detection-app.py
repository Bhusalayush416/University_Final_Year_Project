import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # Convert into the lowercase
    text = nltk.word_tokenize(text)  # break the word in the text

    a = []
    for i in text:
        if i.isalnum():
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        a.append(ps.stem(i))

    return " ".join(a)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title(":green[ :e-mail: _Email Classifier App: Spam or Ham_]")

input_email = st.text_area("Enter The Email You Want To Check", height=150, placeholder="Enter The Email You Want To Check", label_visibility="hidden")

if st.button("Predict", type="secondary"):

    # 1. preprocess
    transformed_email = transform_text(input_email)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_email])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("_Email you have entered is Spam._ :red[:warning: Be careful while using it:warning:]")
        #st.image("Spam.jpg")
    else:
        st.header("Email you have entered is not spam. You are Safe to access the email")
