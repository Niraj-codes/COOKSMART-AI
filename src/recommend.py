import pickle
import pandas as pd
import os

def recommend_recipes(ingredients_input):
    # Load vectorizer and model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path1 = os.path.join(current_dir, 'models', 'tfidf_vectorizer.pkl')
    model_file_path2 = os.path.join(current_dir, 'models', 'knn_model.pkl')

    recipe_file_path1 = os.path.join(current_dir, 'data','processed' ,'processed_dataset.csv')

    with open(model_file_path1, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(model_file_path2, 'rb') as f:
        knn = pickle.load(f)

    # Vectorize user input
    ingredients_vector = vectorizer.transform([ingredients_input])

    # Find the nearest recipes
    distances, indices = knn.kneighbors(ingredients_vector)

    # Load the dataset
    df = pd.read_csv(recipe_file_path1)

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
