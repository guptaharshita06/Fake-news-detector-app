import os
import requests

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

# Links
model_url = "https://drive.google.com/uc?export=download&id=1JuvX-iyLp8yEo4CMMf9J3KEpBe7I_MpB"
vectorizer_url = "https://drive.google.com/uc?export=download&id=1gu4zHYQpJ73QYgfqZjNN3AFb4XgpLzyE"

# Download files automatically
download_file(model_url, "model.pkl")
download_file(vectorizer_url, "vectorizer.pkl")
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# ✅ CORS FIX ADD KAR
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sab allow (dev ke liye)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class News(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Fake News Detector Running"}

@app.post("/predict")
def predict(news: News):
    data = vectorizer.transform([news.text])
    prediction = model.predict(data)

    return {
        "result": "Real News" if prediction[0] == 1 else "Fake News"
    }