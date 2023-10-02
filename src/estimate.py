import surprise.dump
from surprise import accuracy, Dataset, SVD
from surprise.model_selection import train_test_split
from info import FILE_PATH, MODELS_TYPE, get_version
import pickle
import os
from database_service import update_model_metrics, get_db_connection, get_model_by_version, get_ratings_test_data
from surprise import accuracy, Dataset, SVD, Reader
import time
from sklearn.metrics import mean_squared_error

def estimate_svd(connection, version):
    get_model_by_version(connection, MODELS_TYPE.SVD.value, FILE_PATH.MODEL_FILE.value, version)
    with open(FILE_PATH.MODEL_FILE.value, "rb") as file:
        model = pickle.load(file)
    reader = Reader()
    test_data = get_ratings_test_data(connection)
    data = Dataset.load_from_df(test_data, reader)
    trainset, testset = train_test_split(data, test_size=1.00, random_state=42)
    start = time.time()
    predictions = model.test(testset)
    duration = time.time() - start
    os.remove(FILE_PATH.MODEL_FILE.value)
    rmse = accuracy.rmse(predictions)
    update_model_metrics(connection, MODELS_TYPE.SVD.value, version, rmse, duration)

def estimate_knn(connection, version):
    get_model_by_version(connection, MODELS_TYPE.KNN.value, FILE_PATH.MODEL_FILE.value, version)
    with open(FILE_PATH.MODEL_FILE.value, "rb") as file:
        knn = pickle.load(file)
    test_data = get_ratings_test_data(connection)
    os.remove(FILE_PATH.MODEL_FILE.value)
    x = test_data.iloc[:, 0:2].values
    y = test_data.iloc[:, 2:]
    start = time.time()
    y_pred = knn.predict(x)
    duration = time.time() - start
    rmse = mean_squared_error(y, y_pred, squared=False)
    update_model_metrics(connection, MODELS_TYPE.KNN.value, version, rmse, duration)

if __name__ == "__main__":
    connection = get_db_connection()
    version = get_version()
    estimate_svd(connection, version)
    estimate_knn(connection, version)
    connection.close()
