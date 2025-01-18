import pickle
import pandas as pd


def recommend_recipes(ingredients_input):
    # Load vectorizer and model
    with open('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/models/knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)

    # Vectorize user input
    ingredients_vector = vectorizer.transform([ingredients_input])

    # Find the nearest recipes
    distances, indices = knn.kneighbors(ingredients_vector)

    # Load the dataset
    df = pd.read_csv('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/data/processed/processed_dataset.csv')

    # Retrieve the top 5 recipes
    recommended_recipes = df.iloc[indices[0]][['name', 'image_url','ingredients','instructions']].to_dict(orient='records')

    return recommended_recipes

if __name__ == "__main__":
    ingredients_input = input("Enter ingredients (comma separated): ")
    recipes = recommend_recipes(ingredients_input)
    for recipe in recipes:
        print(f"Recipe: {recipe['name']}")
        print(f"Image: {recipe['image_url']}")
        print(f"Ingredients: {recipe['ingredients']}")
        print(f"Instructions: {recipe['instructions']}")
        print('-' * 50)
