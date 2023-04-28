from utils import Utils
from models import Models

if __name__ == '__main__':
    utils = Utils()
    models = Models()

    data = utils.load_from_csv("in/felicidad_org.csv")
    X, y = Utils.features_target(self=NotImplemented, dataset=data, drop_cols=['score', 'rank', 'country'],  y=['score'])

    models.grid_training(X, y)
    print(data.head())