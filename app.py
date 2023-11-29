import pickle

# Load the trained model
with open('naive_bayes_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Import necessary libraries
import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load the 20 Newsgroups dataset
#newsgroups_train = fetch_20newsgroups(subset='train')
#newsgroups_test = fetch_20newsgroups(subset='test')

# Convert the text data into vectors
#vectorizer = TfidfVectorizer()
#vectors_train = vectorizer.fit_transform(newsgroups_train.data)
#vectors_test = vectorizer.transform(newsgroups_test.data)

# Train a Naive Bayes classifier
#clf = MultinomialNB(alpha=.01)
#clf.fit(vectors_train, newsgroups_train.target)

# Streamlit app
st.title('News Document Classification')
document = st.text_input("Enter a news document:")
if st.button('Classify Document'):
    # Convert the document into a vector
    vector = vectorizer.transform([document])
    # Predict the category of the document
    prediction = clf.predict(vector)
    st.write('Category: ', clf.target_names[prediction[0]])
