import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_ml_model():
    """Load the Logistic Regression model and TF-IDF vectorizer."""
    lr_model = joblib.load('models/lr_emotion_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return lr_model, vectorizer

def load_dl_model():
    """Load the LSTM model and tokenizer."""
    lstm_model = load_model('models/lstm_emotion_model.h5', compile=False)
    lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    tokenizer = joblib.load('models/tokenizer.pkl')
    return lstm_model, tokenizer

def predict_ml(text):
    """Predict emotion and confidence using the Logistic Regression model."""
    lr_model, vectorizer = load_ml_model()
    text_vectorized = vectorizer.transform([text])
    probabilities = lr_model.predict_proba(text_vectorized)[0]
    emotion_idx = probabilities.argmax()
    confidence = probabilities[emotion_idx] * 100  # Convert to percentage
    emotion = lr_model.classes_[emotion_idx]
    return emotion, confidence

def predict_dl(text, max_len=100):
    """Predict emotion and confidence using the LSTM model."""
    lstm_model, tokenizer = load_dl_model()
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len)
    predictions = lstm_model.predict(text_pad)[0]  # Get probabilities for all emotions
    emotion_idx = predictions.argmax()
    confidence = predictions[emotion_idx] * 100  # Convert to percentage

    # Manually define the mapping from indices to emotions
    idx_to_emotion = {
        0: "happy",
        1: "excitement",
        2: "sad",
        3: "sarcasm",
        4: "humor",
        5: "anger",
        6: "love",
        7: "surprise",
        8: "abusive",
        9: "fear",
    }

    emotion = idx_to_emotion.get(emotion_idx, "Unknown")
    return emotion, confidence

if __name__ == "__main__":
	while(True):
		print("Select model to test:")
		print("1. Logistic Regression (ML)")
		print("2. LSTM (DL)")
		print("3. Exit")
		choice = input("Enter your choice (1 or 2): ").strip()
		
		if choice == "3":
			break

		text = input("Enter a text to predict its emotion: ").strip()

		if choice == "1":
		    emotion, confidence = predict_ml(text)
		    print(f"Predicted Emotion (ML): {emotion} ({confidence:.2f}%)")
		elif choice == "2":
		    emotion, confidence = predict_dl(text)
		    print(f"Predicted Emotion (DL): {emotion} ({confidence:.2f}%)")
		else:
		    print("Invalid choice. Please select 1 or 2.")

