from langdetect import detect
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load English model (DistilBERT)
DISTILBERT_MODEL_PATH = "../models/distilbert_model"
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model.eval()

# Load IndicBERT model
INDICBERT_MODEL_NAME = "ai4bharat/indic-bert"
indicbert_tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)
indicbert_model = AutoModelForSequenceClassification.from_pretrained(INDICBERT_MODEL_NAME, num_labels=2)  # adjust labels
indicbert_model.eval()

INDIAN_LANG_CODES = ['hi', 'bn', 'ta', 'te', 'ml', 'gu', 'kn', 'mr', 'or', 'pa', 'as']

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def predict_with_model(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

def hybrid_predict(text):
    lang = detect_language(text)
    if lang == 'en':
        pred, confidence = predict_with_model(text, distilbert_tokenizer, distilbert_model)
        model_used = "distilbert"
    elif lang in INDIAN_LANG_CODES:
        pred, confidence = predict_with_model(text, indicbert_tokenizer, indicbert_model)
        model_used = "indicbert"
    else:
        # fallback: treat as English or flag unknown
        pred, confidence = predict_with_model(text, distilbert_tokenizer, distilbert_model)
        model_used = "distilbert"
    return {
        "predicted_label": pred,
        "confidence": confidence,
        "language": lang,
        "model_used": model_used
    }
