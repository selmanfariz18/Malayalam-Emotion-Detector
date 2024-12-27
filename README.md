# Malayalam Emotion Detector

A project to detect emotions in Malayalam texts using Machine Learning (ML) and Deep Learning (DL) models.

## Training Data Format

- The training data must be in `.xlsx` format.
- The Excel file should contain **two columns**:
  1. **Text**: Contains Malayalam sentences.
  2. **Emotion**: Contains one of the following 10 emotions:  
     `"Happy"`, `"Excitement"`, `"Sad"`, `"Sarcasm"`, `"Humour"`, `"Anger"`, `"Love"`, `"Surprise"`, `"Abusive"`, `"Fear"`.
- Save the Excel file inside the `scripts` folder.

---

## How to Train the Models?

### Train the Machine Learning Model (Logistic Regression)

1. Navigate to the `scripts` folder.
2. Run the following command:
   ```bash
   python scripts/train_ml_model.py
   ```

### Train Train the Deep Learning Model (LSTM)

1. Navigate to the `scripts` folder.
2. Run the following command:
   ```bash
   python scripts/train_dl_model.py
   ```

---

## Testing the Trained Models

To test the models, run the following command:

```bash
python test_model.py
```

---

## Important Notes

- The code requires the **TensorFlow** library, which is supported only up to Python version 3.11.  
  Ensure you create a virtual environment with the specified Python version before running the code.
