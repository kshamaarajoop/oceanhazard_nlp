# Hazard_NLP - Training & Inference Guide

This README will walk you through how to train, evaluate, and serve the hazard classification model using the provided source code and data files.

---

## 📁 Project Structure

```
HAZARD_NLP/
│
├── models/                         # Saved model checkpoints and configs
│   └── model_checkpoint/           # Subfolder for model checkpoints
├── processed_data/
│   └── labeled_data.csv            # Cleaned, labeled data for training
├── raw_data/
│   └── social_post.csv             # Raw social posts (unlabeled or source data)
├── src/
│   ├── app.py                      # FastAPI app (serving predictions)
│   ├── train.py                    # Model training script
│   ├── data_prep.py                # Data preprocessing script
│   ├── hybrid_infer.py             # Hybrid inference engine
│   ├── evaluate.py                 # Model evaluation script
│   └── (other utility/eval files)
│
├── input_test.py                   # Example script for API input/output testing
├── myenv/                          # (optional) Python virtual environment
│
├── .gitignore
└── requirements.txt (optional)
```

🚀 Getting Started

1. **Environment Setup**

Activate your Python environment (recommended):

```
source myenv/bin/activate
```
Install requirements (if needed):


transformers torch fastapi uvicorn pandas scikit-learn langdetect pydantic sentencepiece tiktoken

2. **Prepare Your Data**

- Raw social post data: `raw_data/social_post.csv`
- Labeled/processed training data: `processed_data/labeled_data.csv`

To process raw data into labeled format:

```
python src/data_prep.py
```

---

### 3. **Train the Model**

Train the hybrid hazard classifier using the processed data:

```
python src/train.py
```

This will use the data in `processed_data/labeled_data.csv` and save model checkpoints in `models/model_checkpoint/`.

---

### 4. **Evaluate the Model**

After training, evaluate performance (accuracy, recall, etc):

```
python src/evaluate.py
```

---

### 5. **Serve the Model (API)**

To deploy the model as a REST API:

```
uvicorn src.app:app --reload --port 8000
```

Test the API POST endpoint (from another terminal):

```
python input_test.py
```

Or manually with:

```
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"content_text":"Your post text here"}'
```

Or use the interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---
 🚦 Troubleshooting

- **Import or package errors**: Check `requirements.txt` and install missing dependencies.
- **Output not as expected**: Make sure your processed data matches expected schema, and all model files are present in `models/model_checkpoint/`.
- **API Errors**: Check logs for tracebacks, ensure model checkpoints and configs are correct.

---

✨ Tips

- Adjust `max_length`, model type, or hyperparameters in `train.py` and `config.yaml`.
- To train only on English or Hindi, filter data in `data_prep.py`.
- For batch inference/testing, use `input_test.py`.

---

Happy Training! 🚀
```
