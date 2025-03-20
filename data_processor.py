import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config
import imdb

class MovieRecommender:
    def __init__(self, movies_path, ratings_path):
        # Initialize IMDb accessor
        self.ia = imdb.Cinemagoer()
        
        # Load movie and ratings data
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        
        # Prepare data
        self.prepare_data()
    
    def prepare_data(self):
        # Merge movies and ratings
        self.movie_ratings = pd.merge(self.movies, self.ratings, on='movieId')
        
        # Create user-movie rating matrix
        self.user_movie_ratings = self.movie_ratings.pivot_table(
            index='userId', 
            columns='title', 
            values='vote_average'
        )
    
    def get_imdb_movie_details(self, movie_title):
        """
        Fetch movie details from IMDb
        """
        try:
            # Search for the movie
            search_results = self.ia.search_movie(movie_title)
            
            if not search_results:
                return None
            
            # Get the first result and fetch full details
            movie = search_results[0]
            self.ia.update(movie)
            
            # Prepare movie information
            movie_info = {
                'title': movie.get('title', 'N/A'),
                'overview': movie.get('plot outline', 'No overview available'),
                'poster_path': self._get_poster_url(movie),
                'vote_average': movie.get('rating', 0),
            }
            
            return movie_info
        
        except Exception as e:
            print(f"Error fetching movie details: {e}")
            return None
    
    def _get_poster_url(self, movie):
        """
        Generate poster URL for IMDb movies
        """
        try:
            # Try to get poster URL
            if movie.get('full-size cover url'):
                return movie['full-size cover url']
            
            # Fallback to a generic poster URL if available
            return ''
        except:
            return ''
    
    # Rest of the methods remain the same as in the original implementation
    def collaborative_filtering(self, movie_title, n_recommendations=6):
        """
        Collaborative filtering using Pearson correlation
        """
        if movie_title not in self.user_movie_ratings.columns:
            return []
        
        # Calculate movie correlations
        movie_correlations = self.user_movie_ratings.corr(method='pearson')
        
        # Get similar movies
        similar_movies = movie_correlations[movie_title].sort_values(ascending=False)
        similar_movies = similar_movies[similar_movies.index != movie_title]
        
        return similar_movies.head(n_recommendations).index.tolist()
    
    def content_based_filtering(self, movie_title, n_recommendations=6):
        """
        Content-based filtering using cosine similarity
        """
        # Prepare feature matrix
        self.movies['features'] = self.movies['genres'] + ' ' + self.movies['keywords']
        
        # Vectorize features
        vectorizer = CountVectorizer(stop_words='english')
        feature_matrix = vectorizer.fit_transform(self.movies['features'].fillna(''))
        
        # Find movie index
        try:
            movie_index = self.movies[self.movies['title'] == movie_title].index[0]
        except IndexError:
            return []
        
        # Calculate similarity
        similarity_scores = cosine_similarity(feature_matrix[movie_index:movie_index+1], feature_matrix)
        
        # Sort and get top recommendations
        similar_indices = similarity_scores[0].argsort()[::-1][1:n_recommendations+1]
        
        return self.movies.iloc[similar_indices]['title'].tolist()
    
    def get_recommendations(self, movie_title):
        """
        Combine collaborative and content-based recommendations
        """
        collaborative_recs = self.collaborative_filtering(movie_title)
        content_recs = self.content_based_filtering(movie_title)
        
        # Combine and remove duplicates
        all_recommendations = list(dict.fromkeys(collaborative_recs + content_recs))
        #recommend top 6 from the list
        return all_recommendations[-6:]