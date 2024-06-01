
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import snowballstemmer as stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import ssl
import uvicorn
app = FastAPI()


origins = ["http://localhost:5005/admin/comment","http://localhost:5005"]
# origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
from nltk.corpus import stopwords
sw = stopwords.words('turkish')

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
        return "Olumlu"
    elif prediction == 0:
        return "Olumsuz"
    else:
        return "bilinmiyor"

def predict_category(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vectorized)
    return map_to_label(prediction[0])

@app.post("/analyze_sentiment")
async def analyze_sentiment(text: str):
    try:
        category = predict_category(text)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4000)
