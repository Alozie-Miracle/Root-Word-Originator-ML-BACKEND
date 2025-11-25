import os
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import RootRequest
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# === VERCEL-FRIENDLY NLTK SETUP ===
# Use a local folder for NLTK data included in your deployment
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

# Download required NLTK data locally (do this on your machine, not runtime)

# Use a folder in your project
# NLTK_DIR = "./nltk_data"

# Download required data
# nltk.download('wordnet', download_dir=NLTK_DIR)
# nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DIR)
# nltk.download('punkt', download_dir=NLTK_DIR)       # still needed for tokenizer base
# nltk.download('punkt_tab', download_dir=NLTK_DIR)   # specifically for 'punkt_tab'


lemmatizer = WordNetLemmatizer()

# === FASTAPI APP ===
app = FastAPI(title="Root Word Generator", version="1.0.0")

origins = [
    "https://root-word-originator-ml.vercel.app"
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# === HELPER FUNCTIONS ===
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def root_word(text):
    text = text.lower().strip()
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    results = []
    for token, pos in pos_tags:
        wn_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(token, wn_pos)
        results.append({
            "word": token,
            "part_of_speech": pos,
            "root_word": lemma
        })

    if len(results) == 1:
        return results[0]

    return {"original_text": text, "results": results}

# === API ENDPOINTS ===
@app.post("/root")
async def root_endpoint(payload: RootRequest):
    return root_word(payload.word)

@app.get("/")
def read_root():
    return {"message": "Welcome to Word Root Originator API"}
