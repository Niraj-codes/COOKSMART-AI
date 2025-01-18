import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def vectorize_ingredients(input_path, output_vectorizer_path):
    # Load the processed dataset
    df = pd.read_csv(input_path)

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=None, stop_words='english')

    # Fit the vectorizer on the ingredients column
    tfidf_matrix = vectorizer.fit_transform(df['ingredients'])

    # Save the fitted TF-IDF vectorizer
    with open(output_vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print("TF-IDF vectorizer saved successfully!")

if __name__ == "__main__":
    # Check if directory exists, if not create it
    if not os.path.exists('models'):
        os.makedirs('models')

    vectorize_ingredients('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/data/processed/processed_dataset.csv', 'C:/Users/Niraj Prajapati/Desktop/ML PROJECT/models/tfidf_vectorizer.pkl')
