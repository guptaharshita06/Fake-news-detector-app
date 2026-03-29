import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', str(text))
    return text

# Load old data
fake = pd.read_csv("../dataset/Fake.csv")
real = pd.read_csv("../dataset/True.csv")

fake["label"] = 0
real["label"] = 1

# Load new dataset
# Load new dataset
news = pd.read_json("../dataset/news.csv.json", lines=True)

# Create proper text
news["text"] = news["headline"] + " " + news["short_description"]

# Label
news["label"] = 1

print(news.head())
print(news.columns)

# Combine all
data = pd.concat([fake, real, news])

# Clean text
data["text"] = data["text"].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2
)

# Vectorization (UPGRADED)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model (UPGRADED 🔥)
model = PassiveAggressiveClassifier()
model.fit(X_train, y_train)

# Accuracy check
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, pred))

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))