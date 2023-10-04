import numpy as np
import pandas as pd
from database_service import save_rating_data, get_db_connection, save_processed_data, get_movies_raw_data, get_ratings_raw_data, get_keywords_raw_data, get_links_raw_data, save_processed_retrain_data, save_rating_retrain_data
from sklearn.model_selection import train_test_split

def clean_genres(movies):
    movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in eval(x)])
    movies['genres'] = movies['genres'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))
    # print(movies['genres'])
    return movies

def clean_keywords(keywords):
    keywords = keywords.dropna(subset=['keywords'])
    keywords['id'] = pd.to_numeric(keywords['id'])
    keywords['keywords'] = keywords['keywords'].apply(lambda x: [i['name'] for i in eval(x)])
    keywords['keywords'] = keywords['keywords'].apply(lambda x: ' '.join([i.replace(" ", '') for i in x]))
    # print(keywords)
    return keywords

def clean_movies(movies):
    movies = movies.drop([19730, 29503, 35587])
    movies = movies.replace('NaN', np.nan)
    movies = movies.dropna(subset=['vote_average', 'vote_count', 'title'])
    movies = movies.fillna('')


    movies['id'] = pd.to_numeric(movies['id'])
    movies['vote_average'] = pd.to_numeric(movies['vote_average'])
    movies['vote_count'] = pd.to_numeric(movies['vote_count'])
    movies['title'] = movies['title'].astype('str')
    movies['overview'] = movies['overview'].astype('str')

    movies = clean_genres(movies)

    movies['tagline'] = movies['tagline'].fillna('')

    return movies

def merge(movies, keywords, links):
    movies = pd.merge(movies, keywords, on='id')

    movies['description'] = movies['overview'] + ' ' + movies['tagline'] + ' ' + movies['keywords'] + ' ' + movies['genres']
    movies.drop(movies[movies['description'].isnull()].index, inplace=True)

    col = np.array(links['tmdbId'], np.int64)
    links['tmdbId'] = col
    movies.rename(columns={'id': 'tmdbId'}, inplace=True)

    movies = pd.merge(movies, links, on='tmdbId')
    movies.drop(['imdb_id'], axis=1, inplace=True)

    return movies


def preprocess_dataset():
    connection = get_db_connection()

    movies = get_movies_raw_data(connection)
    ratings = get_ratings_raw_data(connection)
    keywords = get_keywords_raw_data(connection)
    links = get_links_raw_data(connection)

    keywords = clean_keywords(keywords)
    movies = clean_movies(movies)
    movies = merge(movies, keywords, links)

    # movies.info()

    movies, drop = train_test_split(movies, test_size=0.5, random_state=42)
    train_data, retrain_data = train_test_split(movies, test_size=0.2, random_state=42)
    train_data.reset_index(inplace=True)
    retrain_data.reset_index(inplace=True)

    save_processed_data(connection, train_data)
    save_processed_retrain_data(connection, retrain_data)

    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    test_data, extra_train_data = train_test_split(test_data, test_size=0.5, random_state=42)
    save_rating_data(connection, train_data, test_data)
    save_rating_retrain_data(connection, extra_train_data)

if __name__ == "__main__":
    preprocess_dataset()
