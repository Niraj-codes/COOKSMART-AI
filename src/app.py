from flask import Flask, request, render_template
from recommend import recommend_recipes

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        recipes = recommend_recipes(ingredients)
        return render_template('index.html', recipes=recipes)
    return render_template('index.html', recipes=None)

if __name__ == "__main__":
    app.run(debug=True)
