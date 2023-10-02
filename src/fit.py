import pickle
import os

from sklearn.model_selection import KFold

from database_service import save_vectorizer, save_model, get_db_connection, \
    get_ratings_train_data, get_ratings_test_data, save_similarity, get_processed_data
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import accuracy, Dataset, SVD, Reader
from info import FILE_PATH, MODELS_TYPE, get_and_increment_version
from surprise.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def fit_SVD(connection, version):
    train_data = get_ratings_train_data(connection)

    print(train_data)

    reader = Reader()
    df = Dataset.load_from_df(train_data, reader)
    kf = KFold(n_splits=5)
    kf.split(df)

    svd = SVD()
    trainset = df.build_full_trainset()
    svd.fit(trainset)

    filename = FILE_PATH.MODEL_FILE.value
    with open(filename, "wb") as file:
        pickle.dump(svd, file)
    save_model(connection, MODELS_TYPE.SVD.value, FILE_PATH.MODEL_FILE.value, version)
    os.remove(FILE_PATH.MODEL_FILE.value)

def fit_kNN(connection, version):
    train_data = get_ratings_train_data(connection)

    knn = KNeighborsRegressor(n_neighbors=10)
    x = train_data.iloc[:, 0:2].values
    y = train_data.iloc[:, 2].values
    knn.fit(x, y)

    filename = FILE_PATH.MODEL_FILE.value
    with open(filename, "wb") as file:
        pickle.dump(knn, file)
    save_model(connection, MODELS_TYPE.KNN.value, FILE_PATH.MODEL_FILE.value, version)
    os.remove(FILE_PATH.MODEL_FILE.value)

def fit_vectorizer(connection, version):
    tfidf = TfidfVectorizer(stop_words='english')
    processed = get_processed_data(connection)
    tfidf_matrix = tfidf.fit_transform(processed['overview'])
    with open(FILE_PATH.VECTORIZER.value, "wb") as file:
        pickle.dump(tfidf, file)

    save_vectorizer(connection, FILE_PATH.VECTORIZER.value, version)
    os.remove(FILE_PATH.VECTORIZER.value)
    return tfidf_matrix

def fit_similarity(connection, version, transformed):
    processed_data = get_processed_data(connection)

    cosine_sim = linear_kernel(transformed, transformed)
    indices = pd.Series(processed_data.index, index=processed_data['title'])
    movies = processed_data[['title', 'vote_average', 'movieId', 'tmdbId']]

    with open(FILE_PATH.SIMILARITY_IND.value, "wb") as file:
        pickle.dump(indices, file)
    with open(FILE_PATH.SIMILARITY_SIM.value, "wb") as file:
        pickle.dump(cosine_sim, file)
    with open(FILE_PATH.SIMILARITY_MOV.value, "wb") as file:
        pickle.dump(movies, file)
    save_similarity(connection, version, FILE_PATH.SIMILARITY_SIM.value, FILE_PATH.SIMILARITY_IND.value, FILE_PATH.SIMILARITY_MOV.value)
    os.remove(FILE_PATH.SIMILARITY_IND.value)
    os.remove(FILE_PATH.SIMILARITY_SIM.value)
    os.remove(FILE_PATH.SIMILARITY_MOV.value)

if __name__ == "__main__":
    connection = get_db_connection()
    current_version = get_and_increment_version()
    transformed = fit_vectorizer(connection, current_version)
    fit_SVD(connection, current_version)
    fit_kNN(connection, current_version)
    fit_similarity(connection, current_version, transformed)
    connection.close()
