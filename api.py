from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import sounddevice as sd
from scipy.io.wavfile import write
import torch
import os
import nltk
import re

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import soundfile as sf

# Set local nltk_data path
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download 
def safe_nltk_download(package):
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package.split("/")[-1], download_dir=nltk_data_path)

safe_nltk_download("tokenizers/punkt")
safe_nltk_download("corpora/stopwords")

# Initialize FastAPI app
app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + processor from local folder
model_dir = os.path.join(os.path.dirname(__file__), "model")
if not os.path.exists(model_dir):
    raise FileNotFoundError("Model folder not found.")

processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)

stop_words = set(stopwords.words("english"))
corpus_text = ""

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})

from fastapi import Request

@app.post("/submit-corpus/")
async def submit_corpus(request: Request):
    form_data = await request.form()
    corpus_text = form_data.get("corpus_text") or ""
    
    global corpus_lines
    normalized = corpus_text.replace('\r\n', '\n').replace('\r', '\n')
    from nltk.tokenize import sent_tokenize
    corpus_lines = sent_tokenize(corpus_text)


    return {"message": f"{len(corpus_lines)} lines loaded from corpus."}


@app.get("/record-and-search/")
def record_and_search():
    duration = 5
    fs = 16000

    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
    except Exception as e:
        return {"error": f"Audio recording failed: {str(e)}"}

    audio = np.squeeze(recording)
    write("user_input.wav", fs, audio)

    try:
        waveform, sr = sf.read("user_input.wav")
    except Exception as e:
        return {"error": f"Failed to read recorded audio: {str(e)}"}

    try:
        input_values = processor(waveform, sampling_rate=fs, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower()
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}

    # Clean and extract keywords
    tokens = word_tokenize(transcription)
    keywords = [t.lower() for t in tokens if t.isalpha() and t.lower() not in stop_words]
    keyword_set = set(keywords)

    # Match based on token overlap
    matched_lines = []
    for line in corpus_lines:
        line_tokens = [t.lower() for t in word_tokenize(line) if t.isalpha()]
        if keyword_set & set(line_tokens):  # intersection not empty
            matched_lines.append(line)
        if len(matched_lines) == 3:
            break

    return {
        "transcription": transcription,
        "keywords": keywords,
        "matched_results": matched_lines or ["No match found."]
    }
