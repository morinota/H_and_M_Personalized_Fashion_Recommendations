from utils.kaggle_api import load_data
from my_class.dataset import DataSet
from train import TrainModel
def main():
    # kaggle APIからデータ読み込み
    load_data()
    # データ加工
    dataset = DataSet()
    dataset.read_data()
    dataset.preprocessing()

    # 学習
    model = TrainModel(dataset=dataset)
    model.train_model()


if __name__ == '__main__':
    main()