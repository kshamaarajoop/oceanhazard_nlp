import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from train import HazardDataset

def main():
    df = pd.read_csv("../processed_data/labeled_data.csv")
    texts = df['clean_text'].tolist()
    labels = df['relevance'].astype(int).tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained("../models/model_checkpoint")
    model = DistilBertForSequenceClassification.from_pretrained("../models/model_checkpoint")

    dataset = HazardDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=16)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k:v for k,v in batch.items() if k != "labels"}
            labels_batch = batch["labels"]
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, axis=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    main()
