import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from seasonal_color_classifier import SeasonalColorClassifier  # Ensure this file exists and is in the correct location

app = Flask(__name__)

# Initialize the classifier
classifier = SeasonalColorClassifier()

# Prepare and fit the classifier model
df = classifier.prepare_training_data()
X = df[['skin_tone', 'hair_color', 'eye_color', 'undertone', 'contrast_level']]
y = df['season']
classifier.fit(X, y)

@app.route('/')
def home():
    """
    Render the home page (index.html).
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the frontend.
    """
    try:
        # Get JSON data from the POST request
        data = request.json

        # Validate input structure
        required_keys = ['skin_tone', 'hair_color', 'eye_color', 'undertone', 'contrast_level']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing required fields in input data'}), 400

        # Create a DataFrame for the input data
        new_person = pd.DataFrame([data])

        # Perform prediction
        predicted_seasons, confidence_scores = classifier.predict_with_confidence(new_person)
        season = predicted_seasons[0]
        confidence_score = confidence_scores[0]

        # Get recommendations
        recommendations = classifier.get_detailed_recommendations(season, confidence_score)

        # Format and return the response
        return jsonify({
            'season': season,
            'confidence': f"{confidence_score * 100:.2f}%",  # Format as percentage
            'recommendations': recommendations
        })

    except Exception as e:
        # Handle errors and provide meaningful feedback
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the environment variable PORT, with a fallback to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
