
from flask import Flask, request, render_template
from recommend import recommend_recipes

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        # Fetch recommended recipes
        recipes = recommend_recipes(ingredients)
        # Render the recipes page with the results
        return render_template('recipes.html', recipes=recipes)
    # Render the input form (index.html) for GET requests
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
