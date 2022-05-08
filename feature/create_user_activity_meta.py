# import 
import numpy as np
import pandas as pd
import os
from math import sqrt
from pathlib import Path
from config import Config

from tqdm import tqdm
tqdm.pandas()
from my_class.dataset import DataSet

import datetime

class CreateUserActivityMeta:

    def __init__(self, dataset:DataSet, transaction_train:pd.DataFrame):
        self.dataset = dataset
        self.df_t = transaction_train

        self.df_t['t_dat'] = pd.to_datetime(self.df_t['t_dat'])
        self.df_t['month'] = self.df_t['t_dat'].dt.month
        self.df_t['year'] = self.df_t['t_dat'].dt.year

        # 2020年だけ抽出
        self.df_t = self.df_t[self.df_t['year']==2020]

        last_ts = self.df_t['t_dat'].max()
        print(f'last day of train_transaction is {last_ts}')

    def get_user_activity_meta(self)->pd.DataFrame:
        self._create_active_status()
        self._create_coldstart_status()
        self._create_frequent_transaction_of_user_in_month()

        self._merge_all_meta_created()
        print(self.result.head())
        
        return self.result

    def _create_active_status(self):
        # まず全ユーザの各月のトランザクション回数をカウントする.
        df_month_avg_item_per_u = self.df_t.groupby(
            by=['customer_id_short', 'month']
            )['price'].count().unstack().reset_index()

        # ユーザのデータとマージする
        df_month_avg_item_per_u = pd.merge(
            self.dataset.df_sub[['customer_id_short']], 
            df_month_avg_item_per_u,
            on='customer_id_short', how='left'
        )

        # トランザクション回数がゼロ回の月の数を算出する
        df_month_avg_item_per_u['num_missing_months'] = (
            df_month_avg_item_per_u.isnull().sum(axis=1)
            )

        # 「各月の購入回数」の欠損値を0埋めする
        df_month_avg_item_per_u = df_month_avg_item_per_u.fillna(0)

        # 'lastest_inactive_months'カラムを作る
        def cal_inactive_months(x:pd.Series):
            """Applyメソッド用。

            Parameters
            ----------
            x : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            if x[9] > 0:
                # 9月に購入有りのユーザ
                return 0
            elif x[9] == 0 and x[8] > 0:
                # 9月に購入なし、8月に購入有りのユーザ
                return 1
            elif x[9] == 0 and x[8] == 0 and x[7] > 0:
                # 8,9月に購入なし、7月に購入有りのユーザ
                return 2
            elif x[9] == 0 and x[8] == 0 and x[7] == 0:
                # 直近の連続3ヶ月以上の非アクティブユーザーは、3
                return 3
                
            else:
                # その他なんてある？？漏れなくない？
                return 4

        df_month_avg_item_per_u['lastest_inactive_months'] = (
            df_month_avg_item_per_u.apply(lambda x: cal_inactive_months(x), axis=1)
        )

        # 'active_status'カラムを作る
        def create_active_status(x:pd.Series):
            if x['num_missing_months'] >= 9:
                return 'inactive_in_year'
            elif x['lastest_inactive_months'] == 3:
                return 'inactive_in_3_months_or_more'
            elif x['lastest_inactive_months'] == 2:
                return 'inactive_in_2_months'
            elif x['lastest_inactive_months'] == 1:
                return 'inactive_in_1_month'
            else:
                # 基本的にはx['lastest_inactive_months'] == 0のユーザ?
                return 'active'

        df_month_avg_item_per_u['active_status'] = (
            df_month_avg_item_per_u.apply(func=create_active_status, axis=1)
        )

        # ユーザのコールドスタートを評価するメタデータ
        self.df_active_user = df_month_avg_item_per_u[['customer_id_short', 'num_missing_months', 'lastest_inactive_months', 'active_status']].copy()


    def _create_coldstart_status(self):
        # まず、トランザクションの数を数える。
        # トランザクション数が10未満のユーザは、コールドスタートユーザと呼ぶ。
        # このようなユーザーは、データが少なすぎて正しいレコメンデーションができないユーザー。
        df_avg_item_per_u = self.df_t.groupby(['customer_id_short'])['price'].count().reset_index()
        df_avg_item_per_u.columns = ['customer_id_short', 'num_transactions']

        # 2020年にトランザクションがないユーザーがいる。
        # これをデータフレームに追加する
        # ＋トランザクション数を0としてラベル付けする。
        df_avg_item_per_u = pd.merge(
            self.dataset.df_sub[['customer_id_short']], 
            df_avg_item_per_u,
            on='customer_id_short', how='left'
        )
        df_avg_item_per_u = df_avg_item_per_u.fillna(0)

        # 2020年のトランザクション数が10未満のユーザをコールドスタートユーザと見なす
        def create_cold_start_status(x:pd.Series):
            if x['num_transactions'] >= Config.borderline_cold_start_user:
                return 'non_cold_start'
            elif x['num_transactions'] < Config.borderline_cold_start_user:
                return 'cold_start'

        df_avg_item_per_u['cold_start_status'] = (
            df_avg_item_per_u.apply(func=create_cold_start_status, axis=1)
        )
        # インスタンス変数として保存
        self.df_coldstart_user = df_avg_item_per_u.copy()

    def _create_frequent_transaction_of_user_in_month(self):

        df_month_avg_item_per_u = self.df_t.groupby(['customer_id_short', 'month'])['price'].count().unstack().reset_index()

        def find_active_month(x):
            """applyメソッド用
            """
            float_x = x.values[1:].astype(float)
            return float_x[~np.isnan(float_x)]
        df_month_avg_item_per_u['transactions_in_active_month'] = df_month_avg_item_per_u.apply(
            lambda x: find_active_month(x), axis=1)

        df_month_avg_item_per_u['mean_transactions_in_active_month'] = df_month_avg_item_per_u.apply(
            lambda x: x['transactions_in_active_month'].mean(), axis=1)

        self.df_transaction_frequent = df_month_avg_item_per_u[['customer_id_short', 'mean_transactions_in_active_month']].copy()

    def _merge_all_meta_created(self):

        self.result = pd.merge(
            self.df_active_user, self.df_coldstart_user,
            on='customer_id_short', how='outer'
            )
        self.result = pd.merge(
            self.result, self.df_transaction_frequent, 
            on='customer_id_short', how='outer'
        )
        del self.df_active_user, self.df_coldstart_user, self.df_transaction_frequent

        return self.result