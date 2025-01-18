from sklearn.neighbors import NearestNeighbors
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(input_file):
    # Load processed data
    df = pd.read_csv(input_file)

    # Load vectorized ingredients
    with open('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(df['ingredients'])

    # Initialize and train the KNN model
    knn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
    knn.fit(X)

    # Save the trained model
    with open('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/models/knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)

if __name__ == "__main__":
    train_model('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/data/processed/processed_dataset.csv')
