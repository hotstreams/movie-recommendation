import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from database_service import get_db_connection, get_vectorizer_from_db, \
    get_data_from_table, get_model_by_version, save_vectorizer, save_model, get_ratings_retrain_data, \
    get_processed_retrain_data, get_processed_data, save_similarity
from info import FILE_PATH, MODELS_TYPE, get_version, get_and_increment_version
import os
import pickle
from surprise import Dataset, Reader
from surprise.model_selection import KFold

def train_svd(connection, version, data):
    filename = FILE_PATH.MODEL_FILE.value
    get_model_by_version(connection, MODELS_TYPE.SVD.value, filename, version)
    with open(filename, "rb") as file:
        model = pickle.load(file)
    os.remove(filename)

    reader = Reader()
    df = Dataset.load_from_df(data, reader)
    kf = KFold(n_splits=5)
    kf.split(df)

    trainset = df.build_full_trainset()
    model.fit(trainset)

    with open(filename, "wb") as file:
        pickle.dump(model, file)
    save_model(connection, MODELS_TYPE.SVD.value, filename, version + 1)
    os.remove(filename)

def train_knn(connection, version, data):
    filename = FILE_PATH.MODEL_FILE.value
    get_model_by_version(connection, MODELS_TYPE.KNN.value, filename, version)
    with open(filename, "rb") as file:
        model = pickle.load(file)
    os.remove(filename)

    x = data.iloc[:, 0:2].values
    y = data.iloc[:, 2].values

    model.fit(x, y)

    with open(filename, "wb") as file:
        pickle.dump(model, file)
    save_model(connection, MODELS_TYPE.KNN.value, filename, version + 1)
    os.remove(filename)

def train_vectorizer(connection, version, extra_train):
    processed_data = get_processed_data(connection)
    data = processed_data._append(extra_train)

    filename = FILE_PATH.VECTORIZER.value
    get_vectorizer_from_db(connection, filename, version)
    with open(filename, "rb") as file:
         vectorizer = pickle.load(file)
    os.remove(filename)

    vectorizer = TfidfVectorizer(stop_words='english')
    transformed = vectorizer.fit_transform(data['overview'])

    filename = FILE_PATH.VECTORIZER.value
    with open(filename, "wb") as file:
        pickle.dump(vectorizer, file)
    save_vectorizer(connection, filename, version + 1)
    os.remove(filename)
    return transformed
def train_similarity(connection, version, extra_train, vectorized_data):
    processed_data = get_processed_data(connection)
    data = processed_data._append(extra_train)
    data.reset_index(inplace=True)
    cosine_sim = linear_kernel(vectorized_data, vectorized_data)
    indices = pd.Series(data.index, index=data['title'])
    movies = data[['title', 'vote_average', 'movieId', 'tmdbId']]

    with open(FILE_PATH.SIMILARITY_IND.value, "wb") as file:
        pickle.dump(indices, file)
    with open(FILE_PATH.SIMILARITY_SIM.value, "wb") as file:
        pickle.dump(cosine_sim, file)
    with open(FILE_PATH.SIMILARITY_MOV.value, "wb") as file:
        pickle.dump(movies, file)
    save_similarity(connection, version + 1, FILE_PATH.SIMILARITY_SIM.value, FILE_PATH.SIMILARITY_IND.value, FILE_PATH.SIMILARITY_MOV.value)
    os.remove(FILE_PATH.SIMILARITY_IND.value)
    os.remove(FILE_PATH.SIMILARITY_SIM.value)
    os.remove(FILE_PATH.SIMILARITY_MOV.value)

if __name__ == "__main__":
    connection = get_db_connection()
    version = get_version()
    extra_ratings_data = get_ratings_retrain_data(connection)
    extra_processed_data = get_processed_retrain_data(connection)
    vectorized = train_vectorizer(connection, version, extra_processed_data)
    train_similarity(connection, version, extra_processed_data, vectorized)
    train_svd(connection, version, extra_ratings_data)
    train_knn(connection, version, extra_ratings_data)
    get_and_increment_version()
    connection.close()