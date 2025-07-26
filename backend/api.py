from flask import Flask, request, jsonify
import pickle
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Initialize NLTK and PorterStemmer
nltk.download('punkt_tab', quiet=True)
stemmer = PorterStemmer()

# Load intents (abbreviated for brevity)
intents = {
    'intents': [
        {'tag': 'abdominal_pain', 'patterns': ['I have abdominal pain', 'My abdomen hurts']},
        {'tag': 'abnormal_menstruation', 'patterns': ['I have a heavy period']},
        {'tag': 'acidity', 'patterns': ['I have acid reflux']},
        # ... (add all intents from your notebook)
        {'tag': 'red_spots_over_body', 'patterns': ['I have red spots on my body']}
    ]
}

# Prepare vocabulary and tags
all_words = []
tags = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
all_words = sorted(set([stemmer.stem(w.lower()) for w in all_words if w not in ['?', '!', '.', ',']]))

# Bag of words function
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Load the trained model
with open('models/fitted_model.pickle2', 'rb') as modelFile:
    model = pickle.load(modelFile)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('symptoms')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    tokens = nltk.word_tokenize(symptoms)
    bow = bag_of_words(tokens, all_words)
    prediction = model.predict(bow.reshape(1, -1))[0]
    return jsonify({'prediction': prediction, 'symptoms': symptoms})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)