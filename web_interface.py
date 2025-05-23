
import streamlit as st
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_text, tokenizer, fit_tokenizer
from tensorflow.keras.datasets import imdb

(_, _), (X_test, _) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
index_word = {v+3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"

texts = [" ".join([index_word.get(i, '?') for i in x]) for x in X_test[:1000]]
fit_tokenizer(texts)

model = load_model("lstm_sentiment_model.h5")

st.title("Sentiment Analysis with LSTM")
user_input = st.text_input("Enter a sentence:")

if user_input:
    x = preprocess_text(user_input)
    pred = model.predict(x)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    st.write(f"Sentiment: **{sentiment}**")
