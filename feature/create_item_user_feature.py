import sys

import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from my_class.dataset import DataSet
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Union
import os

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

# 特徴量生成のベースとなるクラス
class User_and_itemFeatures(ABC):
    @abstractmethod
    def get(self) -> pd.DataFrame:
        """
        customer_id -> features
        """
        pass

class AggrFeatures(User_and_itemFeatures):
    """
    トランザクションログをベースに、ユーザ毎のNumericalデータの特徴量作成。
    Numericalデータとして使えそうなカラムはPriceと
    basic aggregation features(min, max, mean and etc...)
    """

    def __init__(self, dataset:DataSet, transactions_df: pd.DataFrame):
        self.dataset = dataset
        

    def get(self):
if __name__ == '__main__':
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    # dataset.read_data(c_id_short=True)
    dataset.read_data_sampled()

    # データをDataFrame型で読み込み
    df_transaction = dataset.df
    df_sub = dataset.df_sub  # 提出用のサンプル
    dfu = dataset.dfu  # 各顧客の情報(メタデータ)
    dfi = dataset.dfi  # 各商品の情報(メタデータ)