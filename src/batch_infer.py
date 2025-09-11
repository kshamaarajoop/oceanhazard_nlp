import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "../models/model_checkpoint"
INPUT_CSV = "../raw_data/large_social_posts.csv"
OUTPUT_CSV = "../output/predictions.csv"
BATCH_SIZE = 32

def predict_batch(texts, model, tokenizer):
    model.eval()
    preds = []
    probs_list = []
    
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)
            batch_preds = torch.argmax(probs, dim=1).tolist()
            
            preds.extend(batch_preds)
            probs_list.extend(probs.tolist())
    
    return preds, probs_list

def main():
    df = pd.read_csv(INPUT_CSV)
    texts = df["content_text"].tolist()
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    
    preds, probs = predict_batch(texts, model, tokenizer)
    
    df["predicted_label"] = preds
    df["confidence"] = [prob[p] for prob, p in zip(probs, preds)]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
