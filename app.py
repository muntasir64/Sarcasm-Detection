from flask import Flask, render_template, request, send_file, redirect, url_for
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from textblob import TextBlob
from PIL import Image
import pytesseract
import pandas as pd
import io
import sqlite3
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup
from newspaper import Article


app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- LOAD BERT MODEL ----------------
MODEL_PATH = "models/bert_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Optional: Set device (CPU safe)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sarcasm TEXT,
            sentiment TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_to_db(text, sarcasm, sentiment, confidence):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (text, sarcasm, sentiment, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (text, sarcasm, sentiment, confidence,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# ---------------- BERT PREDICTION FUNCTION ----------------
def predict_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item() * 100
    prediction = torch.argmax(probs).item()

    return prediction, round(confidence, 2)

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    prediction = confidence = sentiment = None
    input_text = ""
    error = None
    bulk_results = None
    summary = None

    if request.method == "POST":

        input_text = request.form.get("text")
        url_input = request.form.get("url")
        uploaded_file = request.files.get("file")

        # -------- URL INPUT --------
        if url_input and url_input.strip() != "":
            try:
                article = Article(url_input)
                article.download()
                article.parse()

                extracted_text = article.text

                if len(extracted_text.strip()) == 0:
                    error = "Could not extract readable content from this URL."
                else:
                    input_text = extracted_text[:2000]
            except Exception as e:
                error = f"Error fetching URL: {str(e)}"

        # -------- BULK CSV --------
        elif uploaded_file and uploaded_file.filename.endswith(".csv"):
            try:
                stream = io.StringIO(uploaded_file.stream.read().decode("UTF8"))
                df = pd.read_csv(stream)

                if "text" not in df.columns:
                    error = "CSV must contain a column named 'text'."
                else:
                    bulk_results = []
                    sarcastic_count = positive_count = negative_count = neutral_count = 0

                    for sentence in df["text"]:

                        result, conf = predict_bert(sentence)
                        sarcasm_label = "Sarcastic" if result == 1 else "Not Sarcastic"

                        polarity = TextBlob(sentence).sentiment.polarity
                        if polarity > 0.1:
                            sentiment_label = "Positive"
                            positive_count += 1
                        elif polarity < -0.1:
                            sentiment_label = "Negative"
                            negative_count += 1
                        else:
                            sentiment_label = "Neutral"
                            neutral_count += 1

                        if sarcasm_label == "Sarcastic":
                            sarcastic_count += 1

                        save_to_db(sentence, sarcasm_label,
                                   sentiment_label, conf)

                        bulk_results.append({
                            "sentence": sentence,
                            "sarcasm": sarcasm_label,
                            "sentiment": sentiment_label,
                            "confidence": conf
                        })

                    total = len(df)

                    summary = {
                        "total": total,
                        "sarcastic": sarcastic_count,
                        "positive": positive_count,
                        "negative": negative_count,
                        "neutral": neutral_count
                    }

            except Exception as e:
                error = f"Error reading CSV file: {str(e)}"

        # -------- SINGLE INPUT / FILE --------
        else:

            if uploaded_file and uploaded_file.filename != "":
                filename = uploaded_file.filename.lower()
                try:
                    if filename.endswith(".txt"):
                        input_text = uploaded_file.read().decode("utf-8")

                    elif filename.endswith((".png", ".jpg", ".jpeg")):
                        image = Image.open(uploaded_file)
                        input_text = pytesseract.image_to_string(image)

                except Exception as e:
                    error = f"Error reading uploaded file: {str(e)}"

            if input_text and not error:

                result, confidence = predict_bert(input_text)
                prediction = "Sarcastic" if result == 1 else "Not Sarcastic"

                polarity = TextBlob(input_text).sentiment.polarity
                if polarity > 0.1:
                    sentiment = "Positive"
                elif polarity < -0.1:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                save_to_db(input_text, prediction, sentiment, confidence)

    return render_template(
        "detect.html",
        prediction=prediction,
        confidence=confidence,
        sentiment=sentiment,
        input_text=input_text,
        error=error,
        bulk_results=bulk_results,
        summary=summary
    )
# ---------------- HISTORY PAGE ----------------
@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
    conn.close()
    return render_template("history.html", rows=df.to_dict(orient="records"))

# ---------------- DOWNLOAD HISTORY ----------------
@app.route("/download-history")
def download_history():
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query("SELECT * FROM detections", conn)
    conn.close()

    file_path = "detection_history.csv"
    df.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True)

# ---------------- CLEAR HISTORY ----------------
@app.route("/clear-history", methods=["POST"])
def clear_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return redirect(url_for("history"))

# ---------------- OTHER PAGES ----------------
@app.route("/dataset-model")
def dataset_model():
    return render_template("dataset-model.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
