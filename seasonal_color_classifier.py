import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class SeasonalColorClassifier:
    def __init__(self):
        self.skin_encoder = LabelEncoder()
        self.hair_encoder = LabelEncoder()
        self.eye_encoder = LabelEncoder()
        self.undertone_encoder = LabelEncoder()
        self.contrast_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()

        # Placeholder model, will initialize after hyperparameter tuning
        self.model = None

        # Expanded color recommendations with neutral tones
        self.color_recommendations = {
            'Spring': {
                'bestcolors': ['coral', 'peach', 'warm yellow', 'apple green', 'aqua', 'golden brown', 'warm pink', 'light orange', 'cream'],
                'neutralcolors': ['camel', 'ivory', 'warm beige', 'light brown', 'honey'],
                'makeupcolors': ['peachy pink lipstick', 'warm bronze eyeshadow', 'coral blush', 'apricot lip gloss'],
                'jewelry': ['gold', 'copper', 'bronze', 'rose gold'],
                'avoid': ['black', 'navy', 'dark gray', 'cool blues', 'purple', 'icy tones']
            },
            'Summer': {
                'bestcolors': ['soft pink', 'powder blue', 'gray', 'mauve', 'lavender', 'rose', 'soft green', 'light purple', 'misty blue'],
                'neutralcolors': ['cool gray', 'soft white', 'light navy', 'dove gray'],
                'makeupcolors': ['rose pink lipstick', 'mauve eyeshadow', 'soft pink blush', 'plum lip gloss'],
                'jewelry': ['silver', 'platinum', 'white gold', 'pearl'],
                'avoid': ['orange', 'bright yellow', 'neon shades']
            },
            'Autumn': {
                'bestcolors': ['rust', 'olive green', 'burnt orange', 'brown', 'gold', 'deep red', 'warm teal', 'mustard yellow', 'pumpkin'],
                'neutralcolors': ['warm taupe', 'camel', 'cream', 'khaki'],
                'makeupcolors': ['burnt orange lipstick', 'olive eyeshadow', 'terra cotta blush', 'brick red lip gloss'],
                'jewelry': ['gold', 'bronze', 'copper', 'antique gold'],
                'avoid': ['pink', 'pure white', 'icy tones', 'cool grays']
            },
            'Winter': {
                'bestcolors': ['pure white', 'black', 'royal blue', 'emerald', 'true red', 'icy pink', 'midnight blue', 'cool magenta', 'icy lavender'],
                'neutralcolors': ['charcoal', 'cool gray', 'cool taupe', 'steel blue'],
                'makeupcolors': ['true red lipstick', 'icy blue eyeshadow', 'cool pink blush', 'fuchsia lip gloss'],
                'jewelry': ['silver', 'platinum', 'cool metals', 'diamonds'],
                'avoid': ['orange', 'warm brown', 'earthy tones', 'gold']
            },
            'Neutral': {
                'bestcolors': ['soft peach', 'light teal', 'moss green', 'rose beige', 'dusty pink', 'warm gray', 'taupe'],
                'neutralcolors': ['ivory', 'neutral beige', 'light gray', 'soft white'],
                'makeupcolors': ['nude lipstick', 'neutral brown eyeshadow', 'peachy blush'],
                'jewelry': ['rose gold', 'white gold', 'pearls'],
                'avoid': ['bright neon shades', 'extreme cool or warm tones']
            }
        }

    def prepare_training_data(self):
        """
        Prepare and return training data.
        """
        data = {
            'skin_tone': ['fair-cool', 'fair-warm', 'medium-cool', 'medium-warm', 'deep-cool', 'deep-warm', 'neutral'] * 20,
            'hair_color': ['blonde', 'red', 'brown', 'black', 'dark-brown', 'golden-brown', 'neutral'] * 20,
            'eye_color': ['blue', 'green', 'brown', 'hazel', 'gray', 'dark-brown', 'neutral'] * 20,
            'undertone': ['cool', 'warm', 'cool', 'warm', 'cool', 'warm', 'neutral'] * 20,
            'contrast_level': ['low', 'medium', 'high', 'medium', 'high', 'medium', 'medium'] * 20,
            'season': ['Summer', 'Spring', 'Winter', 'Autumn', 'Winter', 'Autumn', 'Neutral'] * 20
        }
        return pd.DataFrame(data)

    def fit(self, X, y):
        """
        Train the classifier with hyperparameter tuning using GridSearchCV.
        """
        # Encode features
        X_encoded = pd.DataFrame({
            'skin_tone': self.skin_encoder.fit_transform(X['skin_tone']),
            'hair_color': self.hair_encoder.fit_transform(X['hair_color']),
            'eye_color': self.eye_encoder.fit_transform(X['eye_color']),
            'undertone': self.undertone_encoder.fit_transform(X['undertone']),
            'contrast_level': self.contrast_encoder.fit_transform(X['contrast_level'])
        })
        y_encoded = self.season_encoder.fit_transform(y)

        # Hyperparameter grid for RandomForestClassifier
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_encoded, y_encoded)

        # Set the best estimator as the model
        self.model = grid_search.best_estimator_
        return self

    def predict_with_confidence(self, X):
        """
        Predict with confidence scores.
        """
        X_encoded = pd.DataFrame({
            'skin_tone': self.skin_encoder.transform(X['skin_tone']),
            'hair_color': self.hair_encoder.transform(X['hair_color']),
            'eye_color': self.eye_encoder.transform(X['eye_color']),
            'undertone': self.undertone_encoder.transform(X['undertone']),
            'contrast_level': self.contrast_encoder.transform(X['contrast_level'])
        })
        predictions = self.model.predict(X_encoded)
        probabilities = self.model.predict_proba(X_encoded)
        confidence_scores = np.max(probabilities, axis=1)
        seasons = self.season_encoder.inverse_transform(predictions)
        return seasons, confidence_scores

    def get_detailed_recommendations(self, season, confidence_score):
        """
        Get detailed recommendations.
        """
        recommendations = self.color_recommendations.get(season, {})
        if confidence_score >= 0.8:
            confidence_message = "High confidence prediction - these colors should work well for you."
        elif confidence_score >= 0.6:
            confidence_message = "Moderate confidence prediction - try these colors but adjust based on personal preference."
        else:
            confidence_message = "Low confidence prediction - consider trying colors from neighboring seasons as well."

        recommendations['confidence'] = {
            'score': confidence_score,
            'message': confidence_message
        }
        return recommendations