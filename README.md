# AI Disease Detection App

A voice-powered disease detection system that uses machine learning to analyze symptoms and provide preliminary medical assessments.

## Features

- 🎤 **Voice Recording**: Speak your symptoms directly into the microphone
- 🤖 **AI Analysis**: Machine learning model analyzes symptoms and predicts potential conditions
- 💡 **Recovery Tips**: Get personalized recovery recommendations
- 📱 **Web Interface**: User-friendly web application
- 🔒 **Privacy-Focused**: All processing happens locally

## Prerequisites

- Python 3.8 or higher
- Microphone access
- Modern web browser

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the required model files:**
   - `logreg_model.pkl` - Trained logistic regression model
   - `tfidf_vectorizer.pkl` - TF-IDF vectorizer
   - `label_encoder.pkl` - Label encoder for disease categories

## Usage

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Use the application:**
   - Click the microphone button and speak your symptoms
   - Or type your symptoms manually in the text area
   - Click "Analyze Symptoms" to get AI-powered diagnosis
   - Review the predicted condition and recovery tips

## How It Works

1. **Voice Input**: The app records your voice and converts it to text using speech recognition
2. **Text Processing**: The transcribed text is cleaned and preprocessed
3. **ML Analysis**: A trained logistic regression model analyzes the symptoms
4. **Disease Prediction**: The model predicts the most likely condition based on symptom patterns
5. **Recovery Tips**: Personalized recommendations are provided based on the predicted condition

## Supported Conditions

The system can detect and provide tips for various conditions including:
- Heart problems
- Muscle and joint pain
- Headaches and migraines
- Respiratory issues
- Digestive problems
- Skin conditions
- And many more...

## Important Disclaimer

⚠️ **This is an AI-powered preliminary assessment tool. It is NOT a substitute for professional medical diagnosis. Always consult a healthcare professional for proper medical advice and treatment.**

## Technical Details

- **Backend**: Flask web framework
- **ML Model**: Logistic Regression with TF-IDF vectorization
- **Voice Processing**: Google Speech Recognition API
- **Frontend**: Bootstrap 5 with custom JavaScript
- **Data Processing**: scikit-learn, pandas, numpy

## Troubleshooting

### Microphone Issues
- Ensure microphone permissions are granted
- Check if microphone is working in other applications
- Try refreshing the browser page

### Model Loading Issues
- Verify all `.pkl` files are in the project directory
- Check file permissions
- Ensure scikit-learn is properly installed

### Voice Recognition Issues
- Speak clearly and at normal volume
- Reduce background noise
- Try typing symptoms manually if voice recognition fails

## Development

To extend or modify the application:

1. **Add new diseases**: Update the `TIPS_DATABASE` in `app.py`
2. **Improve model**: Retrain with more data and update model files
3. **Enhance UI**: Modify templates and static files
4. **Add features**: Extend the Flask routes and JavaScript functionality

## License

This project is for educational and research purposes. Please use responsibly and always consult medical professionals for health-related decisions.
