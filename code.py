Step 1: Install Required Libraries

pip install numpy tensorflow nltk


Step 2: Sample Dataset

For simplicity, we’ll create a small Q&A dataset as JSON:

# intents.json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
      "responses": ["Hello!", "Hi there!", "Hey! How can I help you?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": ["Goodbye!", "See you later!", "Have a nice day!"]
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "Thanks a lot"],
      "responses": ["You're welcome!", "No problem!", "Anytime!"]
    }
  ]
}

Step 3: Preprocess the Data

import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Prepare data
patterns = []
labels = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        labels.append(intent['tag'])
    if intent['tag'] not in tags:
        tags.append(intent['tag'])

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Encode labels
label_map = {tag: idx for idx, tag in enumerate(tags)}
labels_encoded = np.array([label_map[label] for label in labels])Step 4: Build and Train the Model
from tensorflow.keras.utils import to_categorical

num_classes = len(tags)
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(LSTM(32))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
model.fit(padded_sequences, labels_categorical, epochs=500, verbose=1)
Step 5: Chat Function
def chat():
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        inp = input("You: ").lower()
        if inp == "quit":
            break
        seq = tokenizer.texts_to_sequences([inp])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        pred = model.predict(padded)
        tag_idx = np.argmax(pred)
        tag = tags[tag_idx]

        # Choose random response
        for intent in data['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']
                print("Bot:", np.random.choice(responses))

chat()
