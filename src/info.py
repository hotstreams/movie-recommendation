import enum

class MODELS_TYPE(enum.Enum):
    SVD = "svd"
    KNN = "knn"

class FILE_PATH(enum.Enum):
    MODEL_FILE = "./resources/model"
    VERSION_FILE = "./resources/version"
    VECTORIZER = "./resources/vectorizer"
    SIMILARITY_SIM = "./resources/similarity_data"
    SIMILARITY_IND = "./resources/similarity_indices"
    SIMILARITY_MOV = "./resources/similarity_movies"

def get_and_increment_version():
    with open("./resources/model_version.txt", "r+") as file:
        version = int(file.readline(1)) + 1
        file.seek(0)
        file.write(str(version))
        file.truncate()
        return version

def get_version():
    with open("./resources/model_version.txt", "r") as file:
        return int(file.readline(1))
