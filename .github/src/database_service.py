import pandas as pd
import psycopg2
from psycopg2 import sql
from pandas import DataFrame
import numpy as np

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="postgres"
)

def create_database(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS MOVIES_RAW ("
        "ident SERIAL, "
        "id text, "
        "adult text, "
        "belongs_to_collection text, "
        "budget text,"
        "genres text, "
        "homepage text,"
        "imdb_id text,"
        "original_language text,"
        "original_title text,"
        "overview text,"
        "popularity text,"
        "poster_path text,"
        "production_companies text,"
        "production_countries text,"
        "release_date text,"
        "revenue text,"
        "runtime text,"
        "spoken_languages text,"
        "status text,"
        "tagline text,"
        "title text,"
        "video text,"
        "vote_average text,"
        "vote_count text )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS LINKS_RAW ("
        "movieId bigint primary key, "
        "imdbId bigint, "
        "tmdbId bigint )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS RATINGS_RAW ("
        "id SERIAL, "
        "userId int, "
        "movieId int, "
        "rating int )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS KEYWORDS_RAW ("
        "ident SERIAL, "
        "id int , "
        "keywords text )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS RATINGS_TRAIN_DATA ("
        "id SERIAL, "
        "userId int, "
        "movieId int, "
        "rating int )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS RATINGS_RETRAIN_DATA ("
        "id SERIAL, "
        "userId int, "
        "movieId int, "
        "rating int )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS RATINGS_TEST_DATA ("
        "id SERIAL, "
        "userId int, "
        "movieId int, "
        "rating int )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS MODEL ("
        "id SERIAL, "
        "name text, "
        "model_data bytea, "
        "rmse numeric, "
        "duration numeric, "
        "version int )"
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS PROCESSED_DATA ("
        "id SERIAL, "
        "title text,"
        "vote_average numeric,"
        "movieId numeric,"
        "tmdbId numeric,"
        "overview text,"
        "data text ) "
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS PROCESSED_RETRAIN_DATA ("
        "id SERIAL, "
        "title text,"
        "vote_average numeric,"
        "movieId numeric,"
        "tmdbId numeric,"
        "overview text,"
        "data text ) "
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS VECTORIZER ("
        "id SERIAL, "
        "data bytea,"
        "version int ) "
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS SIMILARITY ("
        "id SERIAL, "
        "sim bytea,"
        "indices bytea,"
        "movies bytea,"
        "version int ) "
    ))
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS BEST_MODEL ("
        "id SERIAL, "
        "version int,"
        "name text,"
        "score numeric ) "
    ))
    cursor.close()
    connection.commit()

if __name__ == "__main__":
    create_database(get_db_connection())

def save_raw_data(connection, movies, ratings, keywords, links):
    cursor = connection.cursor()
    # movies
    for index, row in movies.iterrows():
        cursor.execute(sql.SQL(
            "INSERT INTO MOVIES_RAW ("
            "id, adult, belongs_to_collection, budget, genres, "
            "homepage, imdb_id, original_language, original_title, "
            "overview, popularity, poster_path, production_companies,"
            "production_countries, release_date, revenue, runtime, spoken_languages, "
            "status, tagline, title, video, vote_average, vote_count) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"),
        (
            row['id'], row['adult'], row['belongs_to_collection'], row['budget'], row['genres'],
            row['homepage'], row['imdb_id'], row['original_language'], row['original_title'], row['overview'],
            row['popularity'], row['poster_path'], row['production_companies'], row['production_countries'], row['release_date'],
            row['revenue'], row['runtime'], row['spoken_languages'], row['status'], row['tagline'],
            row['title'], row['video'], row['vote_average'], row['vote_count']
        ))
        "CREATE TABLE RATINGS_RAW ("
        "id SERIAL, "
        "userId int, "
        "movieId int, "
        "rating int, "
        "timestamp TIMESTAMP )"
    # ratings
    for index, row in ratings.iterrows():
        cursor.execute(sql.SQL(
            "INSERT INTO RATINGS_RAW (userId, movieId, rating) values (%s, %s, %s)"),
        (row['userId'], row['movieId'], row['rating']))

    # keywords
    for index, row in keywords.iterrows():
        cursor.execute(sql.SQL(
            "INSERT INTO KEYWORDS_RAW ("
            "id, keywords) values (%s, %s)"),
            (row['id'], row['keywords']))

    # links
    for index, row in links.iterrows():
        cursor.execute(sql.SQL(
            "INSERT INTO LINKS_RAW ("
            "movieId, imdbId, tmdbId) values (%s, %s, %s)"),
            (row['movieId'], row['imdbId'], None if np.isnan(row['tmdbId']) else row['tmdbId']))
    cursor.close()
    connection.commit()

def get_movies_raw_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM MOVIES_RAW ORDER BY ident"))
    result = DataFrame(cursor.fetchall(), index=None, columns= ['ident', 'id', 'adult', 'belongs_to_collection', 'budget', 'genres',
            'homepage', 'imdb_id', 'original_language', 'original_title', 'overview',
            'popularity', 'poster_path', 'production_companies', 'production_countries', 'release_date',
            'revenue', 'runtime', 'spoken_languages', 'status', 'tagline',
            'title', 'video', 'vote_average', 'vote_count'])
    result.drop(columns=result.columns[0], axis=1, inplace=True)
    cursor.close()
    return result

def get_ratings_raw_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM RATINGS_RAW"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['id,', 'userId', 'movieId', 'rating'])
    result.drop(columns=result.columns[0], axis=1, inplace=True)
    cursor.close()
    return result

def get_keywords_raw_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM KEYWORDS_RAW"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['ident,', 'id', 'keywords'])
    result.drop(columns=result.columns[0], axis=1, inplace=True)
    cursor.close()
    return result

def get_ratings_train_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM RATINGS_TRAIN_DATA ORDER BY id"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['id,', 'userId', 'movieId', 'rating'])
    result.drop(columns=result.columns[0], axis=1, inplace=True)
    cursor.close()
    return result

def get_ratings_retrain_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM RATINGS_RETRAIN_DATA ORDER BY id"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['id,', 'userId', 'movieId', 'rating'])
    result.drop(columns=result.columns[0], axis=1, inplace=True)
    cursor.close()
    return result

def get_ratings_test_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM RATINGS_TEST_DATA ORDER BY id"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['id,', 'userId', 'movieId', 'rating'])
    result.drop(columns=result.columns[0], axis=1, inplace=True)
    cursor.close()
    return result


def get_links_raw_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM LINKS_RAW"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['movieId', 'imdbId', 'tmdbId'])
    cursor.close()
    return result

def save_rating_data(connection, train, test):
    cursor = connection.cursor()
    train = train.convert_dtypes()
    test = test.convert_dtypes()
    for index, row in train.iterrows():
        cursor.execute(sql.SQL("INSERT INTO RATINGS_TRAIN_DATA (userId, movieId, rating) values (%s, %s, %s)"),
           (row['userId'], row['movieId'], row['rating']))

    for index, row in test.iterrows():
        cursor.execute(sql.SQL("INSERT INTO RATINGS_TEST_DATA (userId, movieId, rating) values (%s, %s, %s)"),
           (row['userId'], row['movieId'], row['rating']))
    cursor.close()
    connection.commit()

def save_rating_retrain_data(connection, data):
    cursor = connection.cursor()
    data = data.convert_dtypes()
    for index, row in data.iterrows():
        cursor.execute(sql.SQL("INSERT INTO RATINGS_RETRAIN_DATA (userId, movieId, rating) values (%s, %s, %s)"),
                       (row['userId'], row['movieId'], row['rating']))
    cursor.close()
    connection.commit()

def save_model(connection, model, filename, version):
    cursor = connection.cursor()
    with open(filename, "rb") as file:
        cursor.execute(sql.SQL(
            "INSERT INTO MODEL (name, data, version) VALUES (%s, %s, %s)"),
                   (model, psycopg2.Binary(file.read()), version))
    cursor.close()
    connection.commit()

def get_model_by_version(connection, model_name, filename, version):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT model_data from model where version=%s and name=%s"), (version, model_name))
    model_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:
        file.write(model_data)
    cursor.close()

def save_processed_data(connection, data):
    cursor = connection.cursor()
    for index, row in data.iterrows():
        cursor.execute(sql.SQL("INSERT INTO PROCESSED_DATA (title, vote_average, movieId, tmdbId, overview) values (%s, %s, %s, %s, %s)"),
            (row['title'],
             row['vote_average'],
             row['movieId'],
             row['tmdbId'],
             row['description']))
    cursor.close()
    connection.commit()

def get_processed_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select title, vote_average, movieId, tmdbId, overview from PROCESSED_DATA"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['title', 'vote_average', 'movieId', 'tmdbId', 'overview'])
    cursor.close()
    return result

def save_similarity(connection, version, f_sim, f_ind, f_mov):
    cursor = connection.cursor()
    with open(f_ind, "rb") as file:
        ind = psycopg2.Binary(file.read())
    with open(f_sim, "rb") as file:
        sim = psycopg2.Binary(file.read())
    with open(f_mov, "rb") as file:
        mov = psycopg2.Binary(file.read())
    cursor.execute(sql.SQL("INSERT INTO SIMILARITY (version, sim, indices, movies) values (%s, %s, %s, %s)"), (version, sim, ind, mov))
    cursor.close()
    connection.commit()

def get_titles(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select title from PROCESSED_DATA"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['titles'])
    cursor.close()
    return result

def get_similarity(connection, version, f_sim, f_ind, f_mov):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select indices from SIMILARITY where version=%s"), [version])
    indices = cursor.fetchone()[0]
    with open(f_ind, "wb") as file:
        file.write(indices)
    cursor.execute(sql.SQL("select sim from SIMILARITY where version=%s"), [version])
    sim = cursor.fetchone()[0]
    with open(f_sim, "wb") as file:
        file.write(sim)
    cursor.execute(sql.SQL("select movies from SIMILARITY where version=%s"), [version])
    mov = cursor.fetchone()[0]
    with open(f_mov, "wb") as file:
        file.write(mov)
    cursor.close()

def save_processed_retrain_data(connection, data):
    cursor = connection.cursor()
    for index, row in data.iterrows():
        cursor.execute(sql.SQL("INSERT INTO PROCESSED_RETRAIN_DATA (title, vote_average, movieId, tmdbId, overview) values (%s, %s, %s, %s, %s)"),
            (row['title'],
             row['vote_average'],
             row['movieId'],
             row['tmdbId'],
             row['description']))
    cursor.close()
    connection.commit()

def get_processed_retrain_data(connection):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select title, vote_average, movieId, tmdbId, overview from PROCESSED_RETRAIN_DATA"))
    result = DataFrame(cursor.fetchall(), index=None, columns=['title', 'vote_average', 'movieId', 'tmdbId', 'overview'])
    cursor.close()
    return result


def get_best_model(connection, filename):
    cursor = connection.cursor()
    cursor.execute(sql.SQL(
        "select model_data "
        "from (select version, name from best_model order by score ASC limit 1) AS BM join model on BM.name=model.name and BM.version=model.version"))
    model_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:
        file.write(model_data)
    cursor.execute(sql.SQL(
        "select model.name "
        "from (select version, name from best_model order by score ASC limit 1) AS BM join model on BM.name=model.name and BM.version=model.version"))
    name = cursor.fetchone()[0]
    cursor.close()
    return name

def get_model_metrics(connection, current_version):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select name, rmse, duration from model where version=%s"), [current_version])
    result = DataFrame(cursor.fetchall(), index=None)
    cursor.close()
    return result

def update_model_metrics(connection, model_name, current_version, accuracy_score, duration):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("update model set rmse=%s, duration=%s where name=%s and version=%s"), (accuracy_score, duration, model_name, current_version))
    cursor.close()
    connection.commit()


def get_model_by_version(connection, model_name, filename, current_version):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT model_data from model where version=%s and name=%s"), (current_version, model_name))
    model_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:
        file.write(model_data)
    cursor.close()


def get_vectorizer_from_db(connection, filename, current_version):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT data from vectorizer where version=%s"), [current_version])
    vectorizer_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:
        file.write(vectorizer_data)
    cursor.close()


def save_vectorizer(connection, filename, current_version):
    cursor = connection.cursor()
    with open(filename, "rb") as file:
        cursor.execute(sql.SQL("INSERT INTO vectorizer (data, version) VALUES (%s, %s)"),
                       (psycopg2.Binary(file.read()), current_version))
    cursor.close()
    connection.commit()

def save_model(connection, model_name, filename, current_model_version):
    cursor = connection.cursor()
    with open(filename, "rb") as file:
        cursor.execute(sql.SQL("INSERT INTO model (name, model_data, version) VALUES (%s, %s, %s)"),
                       (model_name, psycopg2.Binary(file.read()), current_model_version))
    cursor.close()
    connection.commit()

def save_data(connection, data, table_name):
    cursor = connection.cursor()
    for index, row in data.iterrows():
        cursor.execute(sql.SQL("INSERT INTO {table}(id, title, score, link, summary, published, tickers) values (%s, %s, %s, %s, %s, %s, %s)")
                       .format(table=sql.Identifier(table_name)),
                       (index, row['title'], row['score'], row['link'], row['summary'], row['published'], row['tickers'].replace('[', '{').replace(']', '}')))
    cursor.close()
    connection.commit()


def get_raw_data_from_table(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select * from {table}").format(table=sql.Identifier(table_name)))
    result = DataFrame(cursor.fetchall(), index=None)
    result.set_index(0, inplace=True)
    cursor.close()
    return result


def get_data_from_table(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select * from {table}").format(table=sql.Identifier(table_name)))
    result = DataFrame(cursor.fetchall(), index=None)
    result.set_index(0, inplace=True)
    cursor.close()
    return result


def create_modified_table(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("CREATE TABLE {table} (id integer primary key, score double precision, summary text)").format(table=sql.Identifier(table_name)))
    cursor.close()
    connection.commit()


def save_modified_data(connection, data, table_name):
    cursor = connection.cursor()
    for index, row in data.iterrows():
        cursor.execute(sql.SQL("INSERT INTO {table} (id, score, summary) values(%s, %s, %s)").format(table=sql.Identifier(table_name)), (index, row[1], row[2]))
    cursor.close()
    connection.commit()

def save_best_model(connection, version, name, score):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("INSERT INTO best_model (version, name, score) values (%s, %s, %s)"), (version, name, score))
    cursor.close()
    connection.commit()