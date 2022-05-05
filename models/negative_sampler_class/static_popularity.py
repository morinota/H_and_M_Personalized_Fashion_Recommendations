from datetime import timedelta
from typing import Dict, List, Tuple
from flask import Config
import pandas as pd
from more_itertools import last
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm


class NegativeSamplerStaticPopularity:
    def __init__(self, dataset:DataSet, transaction_train:pd.DataFrame) -> None:
        pass

    def get_negative_record(self,transaction_positive:pd.DataFrame, unique_customer_ids:List, n_negative:int):
        self.negatives_df = pd.DataFrame()
        self.unique_customer_ids = unique_customer_ids

        return self.negatives_df

    def _set_weights_of_sampling(self):
        """各ユニークアイテムの特徴量(=ex. 人気度)をベースにサンプラーの重みを設定
        """

        # 各レコード(アイテム)が抽出される確率
        self.weights_sampler:List[float] = []

    def _create_negative_sampler(self):
        # 各ユーザ毎に繰り返し処理で、Negativeサンプルを付与していく。
        for i, cust_id in tqdm(enumerate(self.unique_customer_ids)):
            pass
        

