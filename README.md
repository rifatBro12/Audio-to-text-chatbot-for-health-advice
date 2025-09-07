# Voice-based Symptom to Disease Predictor

## Overview

This Python project allows users to **speak their symptoms** and get a predicted medical condition along with recovery tips. It leverages **speech recognition**, **TF-IDF vectorization**, and a **trained machine learning model** (Logistic Regression or XGBoost) to classify symptoms into diseases.

## Features

* Real-time **speech input** using microphone.
* **Text preprocessing** (cleaning and lowercasing).
* **TF-IDF vectorization** for keyword extraction.
* Disease prediction using a trained **machine learning model**.
* Display of **tips and recommendations** for predicted diseases.

## Technologies Used

* Python 3
* [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for audio-to-text conversion
* [scikit-learn](https://scikit-learn.org/) for TF-IDF vectorization and Logistic Regression
* [XGBoost](https://xgboost.readthedocs.io/) for optional classifier
* [joblib](https://joblib.readthedocs.io/) for saving and loading models
* Regular expressions (`re`) and `string` for text cleaning

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have **microphone access** and internet connection for Google Speech API.

## Files

* `main.py` : Main script to capture voice and predict disease.
* `logreg_model.pkl` : Trained Logistic Regression model.
* `tfidf_vectorizer.pkl` : TF-IDF vectorizer used for feature extraction.
* `label_encoder.pkl` : Label encoder to convert numeric labels to disease names.

## Usage

Run the main script:

```bash
python main.py
```

### Steps

1. Speak your symptoms clearly into the microphone.
2. The system transcribes your speech.
3. It predicts the disease using the trained model.
4. It displays recovery tips from the built-in tips database.

## Example

```
🎤 Speak your symptoms now...
🗣️ You said: I have pain in my chest and feel weak.

🩺 Predicted Disease: Heart hurts
💡 Recovery Tips: Seek immediate medical help. Avoid stress and exertion.
```

## Notes

* The tips database is limited. For unknown diseases, the user is advised to consult a doctor.
* Models must be trained and saved before running the prediction script.

## Future Enhancements

* Integrate **GUI interface** for better user experience.
* Expand **tips database** for more diseases.
* Add **confidence score** for predictions.
* Enable offline speech recognition.

## License

This project is licensed under the MIT License.
