import os
import requests
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Download function (FIXED)
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)

# ✅ CORRECT LINKS (VERY IMPORTANT)
model_url = "https://drive.google.com/uc?export=download&id=1JuvX-iyLp8yEo4CMMf9J3KEpBe7I_MpB"
vectorizer_url = "https://drive.google.com/uc?export=download&id=1gu4zHYQpJ73QYgfqZjNN3AFb4XgpLzyE"

# ✅ FIRST DOWNLOAD
download_file(model_url, "model.pkl")
download_file(vectorizer_url, "vectorizer.pkl")

# ✅ THEN LOAD
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Request body
class News(BaseModel):
    text: str

# ✅ Routes
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