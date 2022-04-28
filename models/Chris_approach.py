from datetime import timedelta
from typing import Dict, List, Tuple
import pandas as pd
from more_itertools import last
from sqlalchemy import asc
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm
import os

INPUT_DIR = r"input"
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


class ChrisModel:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet, val_week_id: int, k: int = 12) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        # オリジナルのトランザクションログ(アプローチ3で使用)
        self.transaction_train = transaction_train
        # アプローチ1&2で加工していくトランザクションログ
        self.df = transaction_train
        self.val_week_id = val_week_id
        self.k = k

    def _load_pairs_data(self):
        filepath = os.path.join(DRIVE_DIR, 'pairs_cudf.npy')
        self.pairs = np.load(filepath, allow_pickle=True).item()
        print(type(self.pairs))

    def _find_each_customer_last_week_of_purchases(self):
        """各ユーザの最終購入日から、一週間以内のログのみを抽出
        """
        # 各ユーザの最終購入日を得る.
        temp_df = self.df.groupby('customer_id_short')[
            't_dat'].max().reset_index()
        temp_df.columns = ['customer_id_short', 'max_dat']

        # トランザクションログと結合
        self.df = pd.merge(
            left=self.df,
            right=temp_df,
            on=['customer_id_short'], how='left'
        )
        # 各トランザクションで「最終購入日との日数差」のカラムを生成
        self.df['diff_dat'] = (
            self.df['max_dat'] - self.df['t_dat']
        ).dt.days

        # 各ユーザの最終購入日から一週間以内のトランザクションログのみを抽出
        self.df = self.df.loc[
            self.df['diff_dat'] <= 6
        ]
        print(f'train shape is ...{self.df.shape}')

    def preprocessing(self):
        self._load_pairs_data()
        self._find_each_customer_last_week_of_purchases()

    def _recommend_most_often_previously_purchased_items(self):
        # ユーザ毎に、各アイテムの購入回数をカウント
        temp_df = self.df.groupby(['customer_id_short', 'article_id'])[
            't_dat'].agg('count')
        # index化したcustomer_idとarticle_idをカラムに戻す
        temp_df = temp_df.reset_index()
        temp_df.columns = ['customer_id_short', 'article_id', 'ct']

        # ログとマージ
        self.df = pd.merge(
            left=self.df,
            right=temp_df,
            how='left',
            on=['customer_id_short', 'article_id']
        )

        # 「各ユーザ毎の購入回数」が高い、且つt_datが最新の順にソート?
        self.df = self.df.sort_values(
            ['ct', 't_dat'],
            ascending=False)
        # ユーザ×アイテムで、重複したレコードを削除
        self.df.drop_duplicates(
            subset=['customer_id_short', 'article_id'],
            inplace=True
        )
        # 再度ソートしなおす(一応??)
        # 「各ユーザ毎の購入回数」が高い、且つt_datが最新の順にソート?
        self.df.sort_values(
            ['ct', 't_dat'],
            ascending=False, inplace=True
        )

    def _recommend_items_purchased_together(self):
        """「よく一緒に購入される商品」を元にレコメンドを生成する。
        読み込んだpairs(dict)を使用する。
        なお、drop_duplicates()メソッドを使う事で、ユーザが既に購入して、
        _recommend_most_often_previously_purchased_itemsですでにレコメンドされた商品をレコメンドしないように為ている。

        以下は本アプローチの概要
        (1)
        (2)
        (3)
        (4)
        """

        # 「良く一緒に買われるアイテム」1つをログにカラムとして追加
        self.df['paird_item'] = self.df['article_id'].map(
            self.pairs)

        # paird_itemsをレコメンドする
        temp_df = self.df[[
            'customer_id_short', 'paird_item']].copy()
        # paird_items がNaNでないログだけ抽出
        temp_df = temp_df.loc[temp_df['paird_item'].notnull()]
        # 重複を取り除く
        temp_df.drop_duplicates(
            subset=['customer_id_short', 'paird_item'], inplace=True)
        # カラム名を変更
        temp_df.rename(columns={'paird_item': 'article_id'})

        # ペアアイテムのレコメンドは、前回購入したアイテムのレコメンドの後に連結される.
        self.df = self.df[[
            'customer_id_short', 'article_id']]
        # 縦に結合
        self.df = pd.concat(
            objs=[self.df, temp_df],
            axis=0,
            ignore_index=True
        )
        # 重複を取り除く
        self.df.drop_duplicates(
            subset=['customer_id_short', 'article_id'],
            inplace=True
        )

        # レコメンドアイテムのデータ型をString型へ...
        self.df['article_id'] = ' 0' + \
            self.df['article_id'].astype('str')
        # レコメンドを作成
        self.preds = pd.DataFrame(
            # 多分sumで、article_id(文字列)を連結為てる
            self.df.groupby('customer_id_short')[
                'article_id'].sum().reset_index()
        )
        self.preds.columns = ['customer_id_short', 'prediction']

    def _recommend_last_weeks_most_popular_items(self):
        """「前回購入した商品」と「一緒に購入されやすい商品」を
        レコメンドした後、「最も人気のある12個の商品」をレコメンドする。
        """
        # 学習データの最新二週間のみを取り出す
        last_date = self.transaction_train['t_dat'].max()
        init_date = pd.to_datetime(last_date) - timedelta(days=14)
        train = self.transaction_train.loc[
            self.transaction_train['t_dat'] >= pd.to_datetime(init_date)
            ]

        # 2週間で売上上位アイテムを抽出
        self.topk = train['article_id'].value_counts()
        # アイテムidのみを抽出
        self.topk = ' 0' + ' 0'.join(self.topk.index.astype('str')[:self.k])
        

    def create_recommendation(self):
        self._recommend_most_often_previously_purchased_items()
        self._recommend_items_purchased_together()
        self._recommend_last_weeks_most_popular_items()

        # 提出用に整形
        df_sub = self.dataset.df_sub[['customer_id_short', 'customer_id']]

        df_sub = pd.merge(
            left=df_sub,
            right=self.preds,
            on=['customer_id_short'],
            how='left'
        )
        # 欠損値補完
        df_sub.fillna('', inplace=True)
        df_sub['prediction'] = df_sub['prediction'] + self.topk
        df_sub['prediction'] = df_sub['prediction'].str.strip()
        df_sub['prediction'] = df_sub['prediction'].str[:131]

        # 最終的には3つのカラムにする.
        df_sub = df_sub[['customer_id_short', 'customer_id', 'prediction']]

        return df_sub



