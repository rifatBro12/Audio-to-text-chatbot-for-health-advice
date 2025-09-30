import os
import re
import string
import joblib
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import tempfile
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables for models
logreg_model = None
tfidf_vectorizer = None
label_encoder = None

# Tips database for different diseases
TIPS_DATABASE = {
    "Muscle pain": "Rest, apply warm compress, and stay hydrated. Light stretching can help.",
    "Heart hurts": "Seek immediate medical help. Avoid stress and exertion.",
    "Internal pain": "Consult a doctor for proper diagnosis. Avoid spicy foods.",
    "Shoulder pain": "Use ice packs and avoid lifting heavy objects.",
    "Ear ache": "Use ear drops if prescribed. Avoid loud sounds and water.",
    "Feeling cold": "Keep warm and drink hot fluids. Get enough rest.",
    "Knee pain": "Use a knee brace, elevate the leg, and avoid strain.",
    "Infected wound": "Clean the wound, apply antibiotic ointment, and keep it covered.",
    "Hair falling out": "Use mild shampoo, avoid stress, and check diet.",
    "Stomach ache": "Avoid spicy food, drink water, and rest your stomach.",
    "Head ache": "Rest in a dark room, stay hydrated, and avoid screens.",
    "Acne": "Use non-comedogenic skincare and stay hydrated.",
    "Neck pain": "Stretch gently, use proper posture, and apply heat/ice.",
    "Injury from sports": "Use RICE method — Rest, Ice, Compression, Elevation.",
    "Blurry vision": "Avoid screens, take rest, and visit an eye doctor if persists.",
    "Joint pain": "Use anti-inflammatory creams, and stay active but don't overdo it.",
    "Back pain": "Practice good posture, stretch, and use lumbar support.",
    "Skin issue": "Avoid irritants, moisturize, and keep the area clean.",
    "Hard to breath": "Seek immediate medical attention. Stay calm and upright.",
    "Foot ache": "Soak in warm water, elevate foot, and avoid pressure.",
    "Open wound": "Clean properly, cover with sterile dressing, and consult a doctor.",
    "Body feels weak": "Rest, eat nutritious food, and stay hydrated.",
    "Emotional pain": "Talk to someone you trust. Consider professional help.",
    "Feeling dizzy": "Sit or lie down immediately. Avoid quick movements.",
    "Cough": "Stay hydrated, use warm liquids, and rest your throat."
}

def load_models():
    """Load the trained machine learning models"""
    global logreg_model, tfidf_vectorizer, label_encoder
    
    try:
        logreg_model = joblib.load("logreg_model.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def clean_text(text):
    """Clean and preprocess the input text"""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

def predict_disease(symptom_text):
    """Predict disease based on symptom description"""
    if not all([logreg_model, tfidf_vectorizer, label_encoder]):
        return None, "Models not loaded properly"
    
    try:
        # Clean the input text
        cleaned_text = clean_text(symptom_text)
        
        # Transform using TF-IDF vectorizer
        transformed_text = tfidf_vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction_encoded = logreg_model.predict(transformed_text)
        predicted_disease = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get confidence score
        confidence_scores = logreg_model.predict_proba(transformed_text)[0]
        max_confidence = max(confidence_scores)
        
        return predicted_disease, max_confidence
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def get_recovery_tips(disease):
    """Get recovery tips for the predicted disease"""
    return TIPS_DATABASE.get(disease, "Please consult a healthcare professional for proper diagnosis and treatment.")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record_audio():
    """Handle voice recording and transcription"""
    try:
        recognizer = sr.Recognizer()
        
        # Configure recognizer for better accuracy
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.phrase_threshold = 0.3
        
        # Use microphone to record audio
        with sr.Microphone() as source:
            print("Listening for symptoms...")
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Please speak now...")
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=30)
        
        # Convert speech to text with multiple fallback options
        text = None
        error_message = None
        
        # Try Google Speech Recognition first
        try:
            text = recognizer.recognize_google(audio, language='en-US')
            print(f"Transcribed: {text}")
        except sr.UnknownValueError:
            error_message = "Could not understand the audio. Please speak more clearly and try again."
        except sr.RequestError as e:
            print(f"Google recognition failed: {e}")
            # Try alternative recognition service
            try:
                text = recognizer.recognize_sphinx(audio)
                print(f"Transcribed (Sphinx): {text}")
            except Exception as sphinx_error:
                error_message = f"Speech recognition failed. Please try typing your symptoms instead. Error: {str(sphinx_error)}"
        
        if text:
            return jsonify({
                'success': True,
                'transcription': text,
                'message': 'Audio recorded and transcribed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': error_message or 'Could not process audio. Please try again.'
            })
            
    except Exception as e:
        print(f"Recording error: {e}")
        return jsonify({
            'success': False,
            'message': f'Recording error: {str(e)}'
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle disease prediction from text input"""
    try:
        data = request.get_json()
        symptom_text = data.get('symptoms', '').strip()
        
        if not symptom_text:
            return jsonify({
                'success': False,
                'message': 'Please provide symptom description'
            })
        
        # Predict disease
        predicted_disease, confidence = predict_disease(symptom_text)
        
        if predicted_disease is None:
            return jsonify({
                'success': False,
                'message': confidence  # This contains the error message
            })
        
        # Get recovery tips
        tips = get_recovery_tips(predicted_disease)
        
        return jsonify({
            'success': True,
            'predicted_disease': predicted_disease,
            'confidence': round(confidence * 100, 2),
            'recovery_tips': tips,
            'symptoms': symptom_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction error: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    models_loaded = all([logreg_model, tfidf_vectorizer, label_encoder])
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'available_diseases': list(TIPS_DATABASE.keys()) if models_loaded else []
    })

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("Starting AI Disease Detection App...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check if the model files exist.")
        print("Required files: logreg_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl")
