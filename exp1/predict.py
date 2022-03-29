from kaggle import KaggleApi
from preprocessing import DataSet
from implicit.gpu.als import AlternatingLeastSquares


class Prediction:
    def __init__(self, train_model:AlternatingLeastSquares) -> None:
        self.train_model = train_model
# predict process
def predict(dataset:DataSet):
    
    pass



def submit(csv_filepath: str, message: str):
    '''
    Kaggle competitionに結果をSubmitする関数
    '''
    # KaggleApiインスタンスを生成
    api = KaggleApi()
    # 認証を済ませる.
    api.authenticate()

    compe_name = "h-and-m-personalized-fashion-recommendations"
    api.competition_submit(file_name=csv_filepath,
                           message=message, competition=compe_name)


def main():
    # predict something on test dataset

    # submit
    filepath = r'input\sample_submission.csv'
    submit(csv_filepath=filepath, message='submission sample')


if __name__ == '__main__':
    main()
