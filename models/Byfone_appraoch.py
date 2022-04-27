from typing import Dict, List, Tuple
import pandas as pd
from more_itertools import last
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm

INPUT_DIR = r"input"
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


class ByfoneModel:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet, val_week_id: int, k: int = 12) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.transaction_train = transaction_train
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.hyper_params = {}
        self.val_week_id = val_week_id
        self.k = k

    def _calculate_ldbw(self):
        """各トランザクションに対して、最終日との日数差を計算する処理
        """
        # 学習の最終日を取得
        self.last_ts = self.transaction_train['t_dat'].max()
        # 最終日との日数差を計算
        self.transaction_train['ldbw'] = self.transaction_train.apply(
            lambda d: self.last_ts - (self.last_ts - d).floor('7D')
        )

    def _calculate_weekly_sales(self):
        self.weekly_sales = self.transaction_train.drop(
            ['customer_id_short', 'price', 'sales_channel_id', 'week'], axis=1).groupby(['ldbw', 'article_id']).count()
        self.weekly_sales = self.weekly_sales.rename(
            columns={'t_dat': 'count'})

        # 各トランザクションにweekly_salesをマージ
        self.transaction_train = self.transaction_train.join(
            self.weekly_sales, on=['ldbw', 'article_id'])

        pass

    def _calculate_count_targ(self):
        self.weekly_sales = self.weekly_sales.reset_index().set_index('article_id')
        self.last_day = self.last_ts.strftime('%Y-%m-%d')

        # count_targカラムを生成
        self.transaction_train = self.transaction_train.join(
            self.weekly_sales.loc[self.weekly_sales['ldbw']
                                  == self.last_day, ['count']],
            on='article_id', rsuffix="_targ"
        )
        # 欠損を0埋め
        self.transaction_train['count_targ'].fillna(0, inplace=True)

    def _calculate_quotient(self):
        self.transaction_train['quotient'] = self.transaction_train['count_targ'] / \
            self.transaction_train['count']

    def preprocessing(self):
        self._calculate_ldbw()
        self._calculate_weekly_sales()
        self._calculate_count_targ()
        self._calculate_quotient()

    def _recommend_approach_1(self):
        """quotientの各アイテム毎の合計値を算出し、上位k個をgeneral_predとする。
        """
        self.target_sales = self.transaction_train.drop(
            ['customer_id_short', 'price', 'sales_channel_id', 'week'], axis=1
        ).groupby('article_id')['quotient'].sum()
        self.general_pred = self.target_sales.nlargest(self.k).index.tolist()

    def _recommend_approach_2(self):
        """同じアイテムを再度レコメンドする戦略
        """
        self.purchase_dict = {}
        df = self.transaction_train
        for i in tqdm(df.index):
            cust_id = df.at[i, 'customer_id_short']
            art_id = df.at[i, 'article_id']
            t_dat = df.at[i, 't_dat']

            if cust_id not in self.purchase_dict:
                self.purchase_dict[cust_id] = {}

            if art_id not in self.purchase_dict[cust_id]:
                self.purchase_dict[cust_id][art_id] = 0

            x = max(1, (self.last_ts - t_dat).days)

            a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
            y = a / np.sqrt(x) + b * np.exp(-c*x) - d

            value = df.at[i, 'quotient'] * max(0, y)
            self.purchase_dict[cust_id][art_id] += value

    def create_reccomendation(self):
        self._recommend_approach_1()
        self._recommend_approach_2()

        self.df_sub = self.dataset.df_sub[['customer_id_short', 'customer_id']]
        # 両者のレコメンド結果を結合
        pred_list = []
        for cust_id in tqdm(self.df_sub['customer_id_short']):
            if cust_id in self.purchase_dict:
                series = pd.Series(self.purchase_dict[cust_id])
                series = series[series > 0]
                l = series.nlargest(self.k).index.tolist()
                if len(l) < self.k:
                    l = l + self.general_pred[:(self.k-len(l))]
            else:
                l = self.general_pred

            # リストを文字列に変換
            pred_list.append(' '.join([str(x).zfill(10) for x in l]))

        # 提出用に整形
        self.df_sub['prediction'] = pred_list

        # 最終的には3つのカラムにする.
        self.df_sub = self.df_sub[['customer_id_short', 'customer_id', 'prediction']]
        
        return self.df_sub


if __name__ == '__main__':
    pass
