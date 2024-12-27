import pandas as pd
import re
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def split_data(data):
    data['Text'] = data['Text'].apply(preprocess_text)
    return train_test_split(data['Text'], data['Emotion'], test_size=0.2, random_state=42)

