import joblib
from tensorflow.keras.models import load_model

def load_ml_model():
    lr_model = joblib.load('models/lr_emotion_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return lr_model, vectorizer

def load_dl_model():
    lstm_model = load_model('models/lstm_emotion_model.h5')
    tokenizer = joblib.load('models/tokenizer.pkl')
    return lstm_model, tokenizer

