import pandas as pd
import sys
from database_service import get_db_connection, create_database, save_raw_data

def load_dataset():
    movies = pd.read_csv('./resources/movies_metadata.csv')
    ratings = pd.read_csv('./resources/ratings_small.csv')
    keywords = pd.read_csv('./resources/keywords.csv')
    links = pd.read_csv('./resources/links_small.csv')

    connection = get_db_connection()
    create_database(connection)
    save_raw_data(connection, movies, ratings, keywords, links)
    connection.close()

if __name__ == "__main__":
    load_dataset()
