from database_service import get_db_connection, get_model_metrics, save_best_model
from info import get_version

def evaluate_best_model(connection, version):
    model_metrics = get_model_metrics(connection, version)
    result = []
    for index, row in model_metrics.iterrows():
        result.append((row[0], row[1]))
    result.sort(key=lambda x: x[1])
    best = result[0]
    save_best_model(connection, version, best[0], best[1])

if __name__ == "__main__":
    connection = get_db_connection()
    version = get_version()
    evaluate_best_model(connection, version)
    connection.close()