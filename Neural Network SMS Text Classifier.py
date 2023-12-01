import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv('train_data.txt', sep='\t', names=['label', 'message'])
test_data = pd.read_csv('test_data.txt', sep='\t', names=['label', 'message'])

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['message'])
X_test = vectorizer.transform(test_data['message'])

# Define labels
y_train = train_data['label']
y_test = test_data['label']

# Create and train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

def predict_message(message):
    # Vectorize the input message
    message_vector = vectorizer.transform([message])

    # Predict the label (ham or spam) and get the probability
    label = model.predict(message_vector)[0]
    probability = model.predict_proba(message_vector)[0][1]  # Probability of being spam

    # Determine the result based on the probability
    result = "spam" if label == "spam" else "ham"

    return [probability, result]

# Test the model and function
test_messages = [
    "Congratulations! You've won a free vacation!",
    "Hey, what's up? Let's meet tomorrow.",
    "Click the link to claim your prize now!"
]

for message in test_messages:
    prediction = predict_message(message)
    print(f"Message: {message}")
    print(f"Probability: {prediction[0]:.4f}")
    print(f"Result: {prediction[1]}\n")
