from flask import Flask, render_template, request, send_from_directory
from data_processor import MovieRecommender
import os

app = Flask(__name__)

# Initialize recommender
recommender = MovieRecommender('movies.csv', 'ratings.csv')

def get_movie_details(movie_title):
    """
    Fetch comprehensive movie details
    """
    details = recommender.get_imdb_movie_details(movie_title)
    
    if not details:
        raise Exception(f"Movie '{movie_title}' not found")
    
    return details

@app.route('/')
def home():
    # Get movie suggestions (first 100 movies)
    suggestions = recommender.movies['title'].head(100).tolist()
    return render_template('home.html', suggestions=suggestions)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('title')
    
    try:
        # Get movie details for the input movie
        input_movie_details = get_movie_details(movie_title)
        
        if not input_movie_details:
            return render_template('error.html', message=f"Movie '{movie_title}' not found")
        
        # Get recommended movie titles
        recommended_movie_titles = recommender.get_recommendations(movie_title)
        
        # Get details for recommended movies
        recommended_movies = []
        for rec_title in recommended_movie_titles:
            movie_details = get_movie_details(rec_title)
            if movie_details:
                recommended_movies.append(movie_details)
        
        return render_template('recommend.html', 
                               input_movie=input_movie_details, 
                               recommendations=recommended_movies)
    
    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)