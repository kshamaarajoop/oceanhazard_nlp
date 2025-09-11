import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special chars
    return text.lower().strip()

def encode_user_role(role):
    mapping = {'PUBLIC':0, 'VERIFIED_VOLUNTEER':1, 'OFFICIAL':2}
    return mapping.get(role.upper(), 0)

def encode_hazard_type(htype):
    mapping = {
        "HIGH_WAVES":0, "SWELL_SURGE":1, "COASTAL_FLOODING":2,
        "UNUSUAL_TIDE":3, "TSUNAMI_SIGHTING":4, "OTHER":5
    }
    return mapping.get(htype.upper(), 5)

def main():
    df = pd.read_csv("../raw_data/social_post.csv")

    df['clean_text'] = df['content_text'].apply(clean_text)
    df['user_role_enc'] = df['user_role'].apply(encode_user_role)
    df['hazard_type_enc'] = df['hazard_type'].apply(encode_hazard_type)

    labelled_cols = ['clean_text', 'user_role_enc', 'hazard_type_enc', 'sentiment', 'urgency', 'relevance']
    df_processed = df[labelled_cols].dropna()

    df_processed.to_csv("../processed_data/labeled_data.csv", index=False)
    print("Data preprocessed and saved.")

if __name__ == "__main__":
    main()
