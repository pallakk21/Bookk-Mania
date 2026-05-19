from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

popular_df = pickle.load(open(os.path.join(BASE_DIR, 'popular.pkl'), 'rb'))
pt = pickle.load(open(os.path.join(BASE_DIR, 'pt.pkl'), 'rb'))
books = pickle.load(open(os.path.join(BASE_DIR, 'books.pkl'), 'rb'))
similarity_scores = pickle.load(open(os.path.join(BASE_DIR, 'similarity_scores.pkl'), 'rb'))

@app.route('/')
def index():
    image_urls = [
        url.replace("http://", "https://")
        if isinstance(url, str) else ""
        for url in popular_df['Image-URL-M'].values
    ]
    return render_template(
        'index.html',
        book_name=list(popular_df['Book-Title'].values),
        author=list(popular_df['Book-Author'].values),
        image=image_urls,
        votes=list(popular_df['num_ratings'].values),
        rating=list(popular_df['avg_rating'].values)
    )

@app.route('/recommends')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    data = []
    try:
        index = np.where(pt.index == user_input)[0][0]
        similar_items = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:5]
        for i in similar_items:
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            temp_df = temp_df.drop_duplicates('Book-Title')
            title = temp_df['Book-Title'].values[0]
            author = temp_df['Book-Author'].values[0]
            image_url = temp_df['Image-URL-M'].values[0]
            if isinstance(image_url, str):
                image_url = image_url.replace("http://", "https://")
            data.append([title, author, image_url])
    except:
        pass
    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
