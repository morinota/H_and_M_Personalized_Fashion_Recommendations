import pickle
from typing import Dict, List, Tuple
import pandas as pd
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm
import os
import random

INPUT_DIR = r"input"
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


class ContentBaseImage:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.transaction_train = transaction_train
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.hyper_params = {}

    def _load_similarity_data(self):
        self.kNN_json_path = os.path.join(
            DRIVE_DIR, 'input/nearest_neighbor_dictionary.json')
        # 類似度のkNNデータを読み込み
        a_file = open(self.kNN_json_path, "rb")
        self.nearest_neighbor_dictionary: Dict[str, Dict[str, List]]
        self.nearest_neighbor_dictionary = pickle.load(a_file)

    def _create_each_customer_transaction(self):
        self.customer_transactions = self.transaction_train.groupby(
            by="customer_id_short")['article_id'].agg(list).reset_index()

    def _create_dummy_recommendation(self):
        self.most_bought_articles = list(
            (self.transaction_train['article_id'].value_counts()).index)[:12]
        self.most_bought_articles = ' '.join(
            [str(item).zfill(10) for item in self.most_bought_articles])

    def preprocessing(self):
        self._load_similarity_data()
        self._create_each_customer_transaction()
        self._create_dummy_recommendation()

    def _arrange_for_submission(self):
        # submission用に加工
        df_sub = pd.merge(self.dataset.df_sub[['customer_id', 'customer_id_short']], self.customer_transactions, on='customer_id_short', how='left')
        df_sub.rename(columns={"nearest_article_ids":'prediction'}, inplace=True)
        self.df_sub = df_sub[['customer_id', 'customer_id_short', 'prediction']]

    def create_recommendation(self):

        def _add_neighbors(article_ids):
            """
            各ユーザの学習期間における、購入アイテムリストを受け取り、
            kNNをベースにコンテンツベースのレコメンドアイテムのListを返す関数。
            """
            nearest_articles = []
            for article_id in article_ids:

                article_id = str(article_id).zfill(10)
                try:
                    # 各購入アイテム毎に近傍アイテムを取得し、リストに追加
                    nearest_articles = nearest_articles + \
                        self.nearest_neighbor_dictionary[article_id]['nn_article_id'][:5]

                except:
                    continue

            # 2重のリストを一重に変換
            nearest_articles = nearest_articles

            # 12個にする処理.
            if len(nearest_articles) > 12:
                # ランダムに12個を選ぶ.
                nearest_articles = random.sample(nearest_articles, 12)
                # 類似度が高いものを選ぶ
            elif len(nearest_articles) < 12:
                nearest_articles.extend(random.sample(
                    self.most_bought_articles, 12 - len(nearest_articles)))
            return nearest_articles

        # 学習期間内にアイテムを購入した各ユーザに対して、レコメンド
        self.customer_transactions["nearest_article_ids"] = self.customer_transactions.apply(
            lambda row: _add_neighbors(row["article_id"]), axis=1)
        # レコメンド商品のリストを文字列へ
        self.customer_transactions["nearest_article_ids"] = self.customer_transactions["nearest_article_ids"].apply(
            lambda x: " ".join(x))

        self._arrange_for_submission()

        return self.df_sub