import os

from flask import Flask, request
import joblib
from database_service import get_db_connection, get_best_model, get_similarity, get_titles, get_model_by_version
from info import FILE_PATH, MODELS_TYPE, get_version

class Server:
    def __init__(self, m, s, indices, movies, name):
        self.model = m
        self.similarity = s
        self.indices = indices
        self.movies = movies.drop_duplicates()
        self.links = self.movies.set_index('tmdbId')
        self.name = name

    def predict(self, user, movieId):
        if self.name == MODELS_TYPE.SVD.value:
            return self.model.predict(user, movieId).est
        if self.name == MODELS_TYPE.KNN.value:
            return self.model.predict([[user, movieId]])[0]

def get_data(connection):
    name = get_best_model(connection, FILE_PATH.MODEL_FILE.value)
    m = joblib.load(FILE_PATH.MODEL_FILE.value)
    os.remove(FILE_PATH.MODEL_FILE.value)
    get_similarity(connection, get_version(),
                   FILE_PATH.SIMILARITY_SIM.value,
                   FILE_PATH.SIMILARITY_IND.value,
                   FILE_PATH.SIMILARITY_MOV.value)
    ind = joblib.load(FILE_PATH.SIMILARITY_IND.value)
    sim = joblib.load(FILE_PATH.SIMILARITY_SIM.value)
    mov = joblib.load(FILE_PATH.SIMILARITY_MOV.value)
    os.remove(FILE_PATH.SIMILARITY_IND.value)
    os.remove(FILE_PATH.SIMILARITY_SIM.value)
    os.remove(FILE_PATH.SIMILARITY_MOV.value)
    return m, sim, ind, mov, name

connection = get_db_connection()
app = Flask(__name__)
model, sim, ind, mov, name = get_data(connection)
server = Server(model, sim, ind, mov, name)

@app.route('/update')
def update():
    model, sim, ind, mov, name = get_data(connection)
    server.model = model
    server.similarity = sim
    server.indices = ind
    server.movies = mov
    server.name = name
    return 'ok'

@app.route('/set')
def update_with_name():
    if not request.args.get('name'):
        raise RuntimeError('missing name')
    if not request.args.get('version'):
        raise RuntimeError('missing version')

    name = request.args.get('name')
    version = request.args.get('version')
    get_model_by_version(connection, name, FILE_PATH.MODEL_FILE.value, version)
    model = joblib.load(FILE_PATH.MODEL_FILE.value)
    os.remove(FILE_PATH.MODEL_FILE.value)
    get_similarity(connection, version,
                   FILE_PATH.SIMILARITY_SIM.value,
                   FILE_PATH.SIMILARITY_IND.value,
                   FILE_PATH.SIMILARITY_MOV.value)
    ind = joblib.load(FILE_PATH.SIMILARITY_IND.value)
    sim = joblib.load(FILE_PATH.SIMILARITY_SIM.value)
    mov = joblib.load(FILE_PATH.SIMILARITY_MOV.value)
    os.remove(FILE_PATH.SIMILARITY_IND.value)
    os.remove(FILE_PATH.SIMILARITY_SIM.value)
    os.remove(FILE_PATH.SIMILARITY_MOV.value)
    server.model = model
    server.similarity = sim
    server.indices = ind
    server.movies = mov
    server.name = name
    return 'ok'

@app.route('/titles')
def titles():
    return get_titles(connection)['titles'].tolist()

@app.route('/similar')
def get_similar():
    if not request.args.get('title'):
        raise RuntimeError('missing title')
    if request.args.get('title') not in server.indices:
        raise RuntimeError('no such title')

    title = request.args.get('title')
    return get_similar_movies(title, 15)[['title', 'vote_average']].values.tolist()

def get_similar_movies(title, n):
    idx = server.indices[title]
    sim_scores = list(enumerate(server.similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    movies = server.movies.loc[movie_indices]
    return movies

@app.route('/recommended')
def r():
    if not request.args.get('title'):
        raise RuntimeError('missing title')
    if not request.args.get('user'):
        raise RuntimeError('missing user')
    print(server.indices)
    if request.args.get('title') not in server.indices:
        raise RuntimeError('no such title')

    title = request.args.get('title')
    user = int(request.args.get('user'))

    movies = get_similar_movies(title, 50)

    scores = movies['tmdbId'].apply(lambda x: server.predict(user, server.links.loc[x]['movieId']))
    movies['score'] = scores
    movies = movies.sort_values('score', ascending=False)
    return movies.head(15).values.tolist()

if __name__ == '__main__':
    app.run(debug=True)
    connection.close()