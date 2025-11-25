import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import RootRequest
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

app = FastAPI(title="Root Word Generator", version="1.0.0")
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods (GET, POST, OPTIONS, etc)
    allow_headers=["*"],  # allow all headers
)



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

    # Tokenize text (works for one word or whole sentence)
    tokens = word_tokenize(text)

    # Get POS tags for all tokens at once
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

    # If it's a single word, return a single object
    if len(results) == 1:
        return results[0]

    # If it's a sentence, return all processed tokens
    return {"original_text": text, "results": results}

@app.post("/root")
async def root_endpoint(payload: RootRequest):
    return root_word(payload.word)


@app.get("/")
def read_root():
    return {"message": "Welcome to Word Root Originator API"}

