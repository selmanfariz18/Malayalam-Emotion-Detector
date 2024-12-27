import joblib
from preprocess import load_data, split_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
data = load_data('Emotions data.xlsx')
X_train, X_test, y_train, y_test = split_data(data)

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(data['Emotion'].unique()), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert emotions to indices
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(data['Emotion'].unique())}
y_train_idx = y_train.map(emotion_to_idx).values
y_test_idx = y_test.map(emotion_to_idx).values

# Train the model
model.fit(X_train_pad, y_train_idx, epochs=10, validation_data=(X_test_pad, y_test_idx))

# Save the model and tokenizer
model.save('models/lstm_emotion_model.h5')
joblib.dump(tokenizer, 'models/tokenizer.pkl')

