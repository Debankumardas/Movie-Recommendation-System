import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
sns.set(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# ==========================================
# 1. DATA GENERATION (Synthetic Movie Data)
# ==========================================
class MovieDataGenerator:
    """Generates realistic movie dataset for demonstration."""
    
    def __init__(self):
        self.genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Thriller', 'Adventure']
        self.directors = ['Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese', 
                         'Quentin Tarantino', 'James Cameron', 'Ridley Scott', 
                         'David Fincher', 'Denis Villeneuve']
        self.actors = ['Leonardo DiCaprio', 'Tom Hanks', 'Meryl Streep', 'Denzel Washington',
                      'Scarlett Johansson', 'Robert Downey Jr.', 'Emma Stone', 'Brad Pitt',
                      'Jennifer Lawrence', 'Chris Hemsworth', 'Natalie Portman', 'Matt Damon']
        self.movie_titles = [
            'The Dark Knight', 'Inception', 'Interstellar', 'The Shawshank Redemption',
            'Pulp Fiction', 'The Godfather', 'Forrest Gump', 'The Matrix',
            'Gladiator', 'The Prestige', 'Dunkirk', 'Joker', 'Avengers', 'Titanic',
            'Avatar', 'The Lion King', 'Toy Story', 'Finding Nemo', 'Up', 'Inside Out',
            'Parasite', '1917', 'Dune', 'Oppenheimer', 'Barbie', 'Spider-Man',
            'Black Panther', 'Wonder Woman', 'Captain America', 'Thor', 'Iron Man',
            'The Avengers', 'Guardians of the Galaxy', 'Deadpool', 'Logan', 'Venom',
            'The Batman', 'Spider-Man: No Way Home', 'Doctor Strange', 'Black Widow',
            'Eternals', 'Shang-Chi', 'Hawkeye', 'Moon Knight', 'She-Hulk', 'Ms. Marvel'
        ]
        
    def generate_movies(self, n_movies=100):
        """Generate movie data with ratings and metadata."""
        movies = []
        
        for i in range(n_movies):
            # Random selections
            title = self.movie_titles[i % len(self.movie_titles)]
            genre = np.random.choice(self.genres, 2, replace=False)
            director = np.random.choice(self.directors)
            actors = np.random.choice(self.actors, 3, replace=False)
            year = np.random.randint(1990, 2024)
            duration = np.random.randint(90, 180)
            
            # Generate ratings (0-10 scale)
            base_rating = np.random.normal(7.0, 1.5)
            rating = max(0, min(10, base_rating))
            
            # Popularity score (0-100)
            popularity = np.random.randint(30, 100)
            
            # Budget and Revenue
            budget = np.random.randint(10000000, 300000000)
            revenue = int(budget * np.random.uniform(1.5, 5.0))
            
            # Plot summary
            plot_summaries = [
                "A thrilling journey through space and time.",
                "An epic tale of heroism and sacrifice.",
                "A comedy that will make you laugh out loud.",
                "A dramatic story of love and loss.",
                "A sci-fi adventure beyond imagination.",
                "A horror movie that will keep you awake at night.",
                "A romantic journey of two hearts.",
                "A thriller that will keep you on the edge of your seat.",
                "An adventure that will take you to new worlds.",
                "A story of redemption and hope."
            ]
            plot = np.random.choice(plot_summaries)
            
            movie = {
                'Movie_ID': i + 1,
                'Title': title,
                'Genre': ', '.join(genre),
                'Director': director,
                'Actors': ', '.join(actors),
                'Year': year,
                'Duration': duration,
                'Rating': round(rating, 1),
                'Popularity': popularity,
                'Budget': budget,
                'Revenue': revenue,
                'Plot': plot
            }
            movies.append(movie)
        
        return pd.DataFrame(movies)
    
    def generate_user_ratings(self, movies_df, n_users=500):
        """Generate user ratings for movies."""
        user_ratings = []
        
        for user_id in range(1, n_users + 1):
            # Each user rates 10-30 movies
            n_ratings = np.random.randint(10, 30)
            rated_movies = np.random.choice(movies_df['Movie_ID'].values, n_ratings, replace=False)
            
            for movie_id in rated_movies:
                # Rating between 1-10
                rating = np.random.randint(1, 11)
                
                # Get movie details
                movie = movies_df[movies_df['Movie_ID'] == movie_id].iloc[0]
                
                user_ratings.append({
                    'User_ID': user_id,
                    'Movie_ID': movie_id,
                    'Rating': rating,
                    'Genre': movie['Genre'],
                    'Director': movie['Director'],
                    'Year': movie['Year']
                })
        
        return pd.DataFrame(user_ratings)


# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
class MovieDataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_preferences = None
        
    def load_data(self, n_movies=100, n_users=500):
        """Load or generate data."""
        print("Generating Movie Data...")
        generator = MovieDataGenerator()
        self.movies_df = generator.generate_movies(n_movies)
        self.ratings_df = generator.generate_user_ratings(self.movies_df, n_users)
        
        # Data Cleaning
        self.movies_df['Genre'] = self.movies_df['Genre'].apply(lambda x: x.lower())
        self.movies_df['Director'] = self.movies_df['Director'].str.lower()
        self.movies_df['Actors'] = self.movies_df['Actors'].str.lower()
        self.movies_df['Plot'] = self.movies_df['Plot'].str.lower()
        
        # Create combined text for content-based filtering
        self.movies_df['Combined_Text'] = (
            self.movies_df['Genre'] + ' ' + 
            self.movies_df['Director'] + ' ' + 
            self.movies_df['Actors'] + ' ' + 
            self.movies_df['Plot']
        )
        
        return self.movies_df, self.ratings_df
    
    def load_real_data(self, movies_csv, ratings_csv):
        """Load real data from CSV files."""
        try:
            self.movies_df = pd.read_csv(movies_csv)
            self.ratings_df = pd.read_csv(ratings_csv)
            print(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
            
            # Preprocess
            self.movies_df['Genre'] = self.movies_df['Genre'].apply(lambda x: x.lower())
            self.movies_df['Combined_Text'] = (
                self.movies_df['Genre'] + ' ' + 
                self.movies_df['Director'] + ' ' + 
                self.movies_df['Actors'] + ' ' + 
                self.movies_df['Plot']
            )
            
            return self.movies_df, self.ratings_df
        except FileNotFoundError:
            print("CSV files not found. Using synthetic data.")
            return self.load_data()


# ==========================================
# 3. RECOMMENDATION SYSTEM
# ==========================================
class MovieRecommender:
    """Movie recommendation system using multiple approaches."""
    
    def __init__(self, movies_df, ratings_df=None):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.tfidf = None
        self.cosine_sim = None
        self.user_similarity = None
        self.movie_similarity = None
        self.user_item_matrix = None
        
    def content_based_filtering(self):
        """Content-based filtering using TF-IDF and Cosine Similarity."""
        print("Building Content-Based Filter...")
        
        # TF-IDF Vectorization
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df['Combined_Text'])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create index mapping
        self.movies_df['index'] = range(len(self.movies_df))
        self.index_to_movie = dict(zip(self.movies_df['index'], self.movies_df['Title']))
        self.movie_to_index = dict(zip(self.movies_df['Title'], self.movies_df['index']))
        
        print(f"Content-based filter built for {len(self.movies_df)} movies")
        return self.cosine_sim
    
    def collaborative_filtering(self):
        """Collaborative filtering using user-item matrix."""
        if self.ratings_df is None:
            print("No ratings data available for collaborative filtering.")
            return None
        
        print("Building Collaborative Filter...")
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='User_ID', 
            columns='Movie_ID', 
            values='Rating', 
            fill_value=0
        )
        
        # Calculate user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Calculate movie similarity
        self.movie_similarity = cosine_similarity(self.user_item_matrix.T)
        
        print(f"Collaborative filter built for {len(self.user_item_matrix)} users and {len(self.user_item_matrix.columns)} movies")
        return self.user_item_matrix
    
    def get_similar_movies(self, movie_title, top_n=10):
        """Get similar movies based on content."""
        if self.cosine_sim is None:
            self.content_based_filtering()
        
        # Get movie index
        try:
            idx = self.movie_to_index[movie_title]
        except KeyError:
            # Find closest match
            matches = self.movies_df[self.movies_df['Title'].str.contains(movie_title, case=False)]
            if len(matches) > 0:
                idx = matches.index[0]
            else:
                print("Movie not found.")
                return None
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # Exclude the movie itself
        
        # Get movie titles
        movie_indices = [i[0] for i in sim_scores]
        similar_movies = self.movies_df.iloc[movie_indices]
        
        return similar_movies
    
    def recommend_for_user(self, user_id, top_n=10):
        """Recommend movies for a specific user based on collaborative filtering."""
        if self.ratings_df is None:
            print("No ratings data available.")
            return None
        
        # Get user's ratings
        user_ratings = self.ratings_df[self.ratings_df['User_ID'] == user_id]
        
        if len(user_ratings) == 0:
            print(f"No ratings found for User {user_id}. Returning popular movies.")
            return self.get_popular_movies(top_n)
        
        # Get movies user hasn't rated
        rated_movies = set(user_ratings['Movie_ID'].values)
        all_movies = set(self.movies_df['Movie_ID'].values)
        unrated_movies = all_movies - rated_movies
        
        # Calculate predicted ratings
        predictions = []
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_similarities = self.user_similarity[user_idx]
        
        for movie_id in unrated_movies:
            # Get ratings for this movie from similar users
            movie_ratings = self.user_item_matrix[movie_id]
            similar_ratings = movie_ratings * user_similarities
            
            # Weighted average
            if similar_ratings.sum() > 0:
                predicted_rating = similar_ratings.sum() / user_similarities.sum()
                predictions.append({
                    'Movie_ID': movie_id,
                    'Predicted_Rating': predicted_rating
                })
        
        # Sort by predicted rating
        predictions = sorted(predictions, key=lambda x: x['Predicted_Rating'], reverse=True)
        predictions = predictions[:top_n]
        
        # Get movie details
        if predictions:
            movie_ids = [p['Movie_ID'] for p in predictions]
            recommended_movies = self.movies_df[self.movies_df['Movie_ID'].isin(movie_ids)]
            recommended_movies['Predicted_Rating'] = [p['Predicted_Rating'] for p in predictions]
            return recommended_movies
        
        return None
    
    def get_popular_movies(self, top_n=10):
        """Get most popular movies based on ratings."""
        popular = self.ratings_df.groupby('Movie_ID').agg({
            'Rating': ['mean', 'count']
        }).reset_index()
        popular.columns = ['Movie_ID', 'Avg_Rating', 'Num_Ratings']
        popular = popular.sort_values('Num_Ratings', ascending=False)
        
        popular_movies = self.movies_df.merge(popular, on='Movie_ID')
        return popular_movies.head(top_n)
    
    def hybrid_recommendation(self, user_id, top_n=10):
        """Hybrid recommendation combining content-based and collaborative filtering."""
        # Get user's most rated genre
        user_genres = self.ratings_df[self.ratings_df['User_ID'] == user_id]['Genre'].value_counts()
        if len(user_genres) > 0:
            preferred_genre = user_genres.index[0]
        else:
            preferred_genre = 'action'
        
        # Filter movies by preferred genre
        genre_movies = self.movies_df[self.movies_df['Genre'].str.contains(preferred_genre, case=False)]
        
        # Get content-based similarity for these movies
        if len(genre_movies) > 0:
            genre_indices = genre_movies['index'].values
            sim_scores = self.cosine_sim[genre_indices]
            # Get top N indices for each row
            top_similar = np.argsort(sim_scores, axis=1)[:, -top_n:]
            
            # Flatten and get unique movies
            top_indices = top_similar.flatten()
            unique_indices = np.unique(top_indices)
            
            recommended = genre_movies.iloc[unique_indices[:top_n]]
            return recommended
        
        return self.get_popular_movies(top_n)


# ==========================================
# 4. USER INTERFACE & INTERACTION
# ==========================================
class MovieRecommendationInterface:
    """Interactive interface for movie recommendations."""
    
    def __init__(self, recommender):
        self.recommender = recommender
        self.movies_df = recommender.movies_df
        
    def display_movie_info(self, movie_id):
        """Display detailed information about a movie."""
        movie = self.movies_df[self.movies_df['Movie_ID'] == movie_id].iloc[0]
        
        print("\n" + "="*60)
        print(f"🎬 {movie['Title']}")
        print("="*60)
        print(f"📅 Year: {movie['Year']}")
        print(f"🎭 Genre: {movie['Genre']}")
        print(f"🎥 Director: {movie['Director']}")
        print(f"🌟 Actors: {movie['Actors']}")
        print(f"⏱️ Duration: {movie['Duration']} minutes")
        print(f"⭐ Rating: {movie['Rating']}/10")
        print(f"💰 Budget: ${movie['Budget']:,}")
        print(f"💵 Revenue: ${movie['Revenue']:,}")
        print(f"📝 Plot: {movie['Plot']}")
        print("="*60)
    
    def search_movies(self, query):
        """Search for movies by title, genre, or director."""
        query = query.lower()
        results = self.movies_df[
            self.movies_df['Title'].str.contains(query, case=False) |
            self.movies_df['Genre'].str.contains(query, case=False) |
            self.movies_df['Director'].str.contains(query, case=False)
        ]
        return results
    
    def get_user_preferences(self):
        """Get user preferences for recommendations."""
        print("\n" + "="*60)
        print("🎯 MOVIE PREFERENCE SETUP")
        print("="*60)
        
        print("\nAvailable Genres:")
        genres = self.movies_df['Genre'].unique()
        for i, genre in enumerate(genres, 1):
            print(f"  {i}. {genre}")
        
        print("\nAvailable Directors:")
        directors = self.movies_df['Director'].unique()
        for i, director in enumerate(directors, 1):
            print(f"  {i}. {director}")
        
        print("\nPlease enter your User ID (1-500) to load your history:")
        user_id = int(input("Enter your User ID: "))
        return user_id
    
# ==========================================
# 5. MAIN PROGRAM EXECUTION
# ==========================================

def main():

    print("\n========================================")
    print("🎬 MOVIE RECOMMENDATION SYSTEM")
    print("========================================")

    # Load data
    processor = MovieDataProcessor()
    movies_df, ratings_df = processor.load_data()

    # Build recommendation system
    recommender = MovieRecommender(movies_df, ratings_df)

    recommender.content_based_filtering()
    recommender.collaborative_filtering()

    # Interface
    interface = MovieRecommendationInterface(recommender)

    while True:

        print("\n========================================")
        print("📌 MENU")
        print("1. Search Movies")
        print("2. Get Similar Movies")
        print("3. Recommend Movies for User")
        print("4. Show Popular Movies")
        print("5. Exit")
        print("========================================")

        choice = input("Enter your choice: ")

        # Search movie
        if choice == "1":

            query = input("Enter movie title / genre / director: ")
            results = interface.search_movies(query)

            if len(results) > 0:
                print("\nSearch Results:")
                print(results[['Movie_ID','Title','Genre','Year','Rating']].head(10))
            else:
                print("No movies found.")

        # Similar movies
        elif choice == "2":

            title = input("Enter movie title: ")
            similar = recommender.get_similar_movies(title)

            if similar is not None:
                print("\nRecommended Similar Movies:")
                print(similar[['Title','Genre','Year','Rating']])

        # User recommendation
        elif choice == "3":

            user_id = int(input("Enter User ID (1-500): "))

            rec = recommender.recommend_for_user(user_id)

            if rec is not None:
                print("\nMovies Recommended For You:")
                print(rec[['Title','Genre','Year','Rating']])

        # Popular movies
        elif choice == "4":

            popular = recommender.get_popular_movies()

            print("\nPopular Movies:")
            print(popular[['Title','Genre','Year','Rating','Num_Ratings']])

        # Exit
        elif choice == "5":

            print("\nThank you for using the Movie Recommendation System 🎬")
            break

        else:
            print("Invalid choice. Try again.")


# Run Program
if __name__ == "__main__":
    main()