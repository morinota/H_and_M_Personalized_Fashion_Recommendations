from kaggle import KaggleApi
from scripts.dataset import DataSet
import implicit


class TrainModel:
    def __init__(self, dataset: DataSet) -> None:
        self.rating_matrix = dataset.coo_train
        self.hypyr_params = {'factors': 500, 'iterations': 3, 'regularization': 0.01}

    def train_model(self):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.hypyr_params['factors'],
            iterations=self.hypyr_params['iterations'],
            regularization=self.hypyr_params['regularization'],
            random_state=42
        )
        self.model.fit(self.rating_matrix, show_progress=True)

        
    pass
