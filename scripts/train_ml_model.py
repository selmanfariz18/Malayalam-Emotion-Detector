import joblib
from preprocess import load_data, split_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load and preprocess data
data = load_data('Emotions data.xlsx')
X_train, X_test, y_train, y_test = split_data(data)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = lr_model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(lr_model, 'lr_emotion_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

