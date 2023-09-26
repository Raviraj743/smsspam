import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

tfidf=TfidfVectorizer()



ps=PorterStemmer()

def data_preprocess(text):
    text= text.lower()
    text=nltk.word_tokenize(text)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    y=[]
    for i in text:
        if i.isalnum:
            y.append(i)
    text= y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in punc:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf= pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Detection/Classifier")
input_sms=st.text_area("Enter the message or sms")
if st.button("Predict"):

    transform_sms= data_preprocess(input_sms) #preprocess
    vector_input= tfidf.transform([transform_sms]) # Vectorize
    result= model.predict(vector_input)[0]

    if result==1:
        st.header("SPAM")

        video1=open("Spam.mp4","rb")
        st.video(video1 , start_time=1)


    else:
        st.header("NOT SPAM")
        video2=open("NOT SPAM.mp4","rb")
        st.video(video2 , start_time=1)