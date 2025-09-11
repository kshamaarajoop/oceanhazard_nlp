import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def predict(texts):
    tokenizer = DistilBertTokenizerFast.from_pretrained("../models/model_checkpoint")
    model = DistilBertForSequenceClassification.from_pretrained("../models/model_checkpoint")
    model.eval()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.tolist(), probs.tolist()

if __name__ == "__main__":
    texts = ["High waves warning in area", "The high waves are perfect for surfing","जल स्तर तेजी से बढ़ रहा है"]
    preds, probs = predict(texts)
    for t, p, prob in zip(texts, preds, probs):
        print(f"Text: {t}\nPrediction: {p}\nProbabilities: {prob}\n")
