from multiprocessing.spawn import import_main_path
from typing import List, Tuple
import pandas as pd
from dataset import DataSet
from useful_func import iter_to_str
from collections import defaultdict

class LSTM:
    # クラス変数の定義
    INPUT_DIR = r"input"
    DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

    def __init__(self, transaction_train: pd.DataFrame) -> None:
    # インスタンス変数(属性の初期化)
        self.transaction_train = transaction_train

    def preprocessing(self):
        self.transaction_train['t_dat'] = pd.to_datetime