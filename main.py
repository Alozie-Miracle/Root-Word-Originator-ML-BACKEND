import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import RootRequest
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer

# === FASTAPI APP ===
app = FastAPI(title="Root Word Generator", version="1.0.0")

origins = [
    "*"
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lemmatizer = WordNetLemmatizer()

# === HELPER FUNCTIONS ===
def simple_pos(word):
    if word.endswith("ing"):
        return "VERB"
    if word.endswith("ed"):
        return "VERB"
    if word.endswith("s"):
        return "NOUN"
    return "NOUN"



def root_word(text):
    tokens = text.split()

    results = []
    for w in tokens:
        pos = simple_pos(w)
        lemma = lemmatizer.lemmatize(w.lower())

        results.append({
            "word": w,
            "root_word": lemma,
            "part_of_speech": pos,
        })

    return {
        "original_text": text,
        "results": results,
    }

# === API ENDPOINTS ===
@app.post("/root")
async def root_endpoint(payload: RootRequest):
    return root_word(payload.word)

@app.get("/")
def read_root():
    return {"message": "Welcome to Word Root Originator API"}
