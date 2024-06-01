import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
import snowballstemmer as stemmer
import joblib
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Türkçe stopwords listesini indirin
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('turkish')

# Modeli ve diğer önemli dosyaları yükle
model = joblib.load('nb_model_multi.pkl')
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in sw])
    stemmer_instance = stemmer.TurkishStemmer()
    text = ' '.join([stemmer_instance.stemWord(word) for word in text.split()])
    return text

def map_to_label(prediction):
    if prediction == 1:
        return "olumlu"
    elif prediction == 0:
        return "olumsuz"
    else:
        return "bilinmiyor"


def predict_category(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vectorized)
    return map_to_label(prediction[0])

@st.cache_data
def get_category(user_input):
    if user_input:
        category = predict_category(user_input)
        return {"category": category}
    else:
        return {"error": "Lütfen bir yorum girin."}



def main():
    st.title("E-Ticaret Yorum Kategorizasyonu")

    # Kullanıcıdan metin girişi al
    user_input = st.text_area("Lütfen bir yorum girin:", "")

    if st.button("Yorumu Sınıflandır"):
        result = get_category(user_input)

        if "error" in result:
            st.warning(result["error"])
        else:
            st.success(f"Yorumunuz {result['category']} kategorisine aittir.")
if __name__ == "__main__":
    main()
 
