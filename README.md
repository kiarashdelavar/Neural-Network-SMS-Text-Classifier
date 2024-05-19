# Neural-Network-SMS-Text-Classifier

This project involves building a machine learning model to classify SMS messages as either "ham" (normal messages from friends) or "spam" (advertisements or messages from companies). The model will be implemented using a neural network to achieve high accuracy.

Dataset Description:
The SMS Spam Collection dataset will be used for this project. The dataset is already split into training and testing sets, providing a convenient starting point for model development and evaluation.

Objective:

Develop a function called predict_message that takes an SMS message string as input and returns a list. The list will contain two elements:

    1- A number between 0 and 1 indicating the likelihood of the message being "ham" (0) or "spam" (1).
    2- A string "ham" or "spam" based on the predicted likelihood.

Methodology:

    1-Data Import and Preprocessing: Load the dataset and preprocess the text data, including tokenization, vectorization, and normalization.
    2-Model Development: Build and train a neural network model using a suitable deep learning framework such as TensorFlow or PyTorch.
    3-Prediction Function: Implement the predict_message function that uses the trained model to classify new SMS messages.

Function: predict_message:
The predict_message function will process the input message and utilize the trained neural network model to predict whether the message is "ham" or "spam".

Implementation:
   
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # Load and preprocess data
    # Add your data loading and preprocessing code here

    # Build and train the neural network model
    # Add your model building and training code here

    # Define the prediction function
     def predict_message(message):
       # Add your prediction code here
        return
        # Test the function
    print(predict_message("Congratulations! You've won a free ticket to the Bahamas. Text WIN to 12345 to claim."))


Place your data preprocessing, model training, and prediction logic between the provided cells to complete the implementation.

