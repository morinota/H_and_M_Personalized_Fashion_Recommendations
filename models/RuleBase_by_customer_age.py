from datetime import timedelta
from msilib import init_database
from typing import Dict, List, Tuple
import pandas as pd
from more_itertools import last
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm

INPUT_DIR = r"input"
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


class RuleBaseByCustomerAge:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet, val_week_id: int, k: int = 12) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.transaction_train = transaction_train
        self.df_u = dataset.dfu[['customer_id_short', 'age']]
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.hyper_params = {}
        self.val_week_id = val_week_id
        self.k = k

    def _grouping_each_age(self):
        """ユーザを年齢層毎にグルーピング
        """
        self.ageBin = [-1, 19, 29, 39, 49, 59, 69, 119]
        self.df_u['age_bins'] = pd.cut(
            x=self.df_u['age'],
            bins=self.ageBin
        )

    def _check_missing_data_of_age(self):
        """年齢の欠損値の数を確認
        """
        pass

    def _extract_recent_transaction(self):
        """最近(直近三週間)のトランザクションのみを抽出。
        """
        # 最近(直近三週間)のトランザクションのみを抽出。
        last_date = self.transaction_train['t_dat'].max()
        init_date = pd.to_datetime(last_date) - timedelta(days=21)

        self.df_recent_t = self.transaction_train.loc[
            self.transaction_train['t_dat'] >= init_date
        ]

    def _add_age_bin_to_recent_transaction(self):
        """トランザクションデータに年齢層ビンを付与。
        """
        self.df_recent_t = pd.merge(
            left=self.df_recent_t,
            right=self.df_u[['customer_id_short', 'age_bins']],
            on='customer_id_short',
            how='inner'
        )

    def _count_recent_popular_articles_of_each_ages(self):

        # 年齢層毎に、各アイテムの売り上げをカウント
        self.recent_popular_items_with_ages = self.df_recent_t.groupby(
            by=['age_bins', 'article_id']).count().reset_index()
        # カラム名を変更
        self.recent_popular_items_with_ages.rename(
            columns={'customer_id_short': 'counts'},
            inplace=True
        )

        # age_binsのユニーク値のリストを保存
        self.list_UniBins = self.recent_popular_items_with_ages['age_bins'].unique(
        ).tolist()

        pass

    def preprocessing(self):
        self._grouping_each_age()
        self._check_missing_data_of_age()
        self._extract_recent_transaction()
        self._add_age_bin_to_recent_transaction()
        self._count_recent_popular_articles_of_each_ages()

    def _f1_extract_df_customer_each_age_bin(self, unique_age_bin: str):
        """各年齢ビンに該当するdf_customerを抽出する。

        Parameters
        ----------
        unique_age_bin : str
            年齢ビン
        """
        if str(unique_age_bin) == 'nan':
            self.df_u_each_age_bin = self.df_u[self.df_u['age_bins'].isnull()]
        else:
            self.df_u_each_age_bin = self.df_u[self.df_u['age_bins']
                                               == unique_age_bin]

        # age_binsカラムを落とす.
        self.df_u_each_age_bin.drop(['age_bins'], axis=1, inplace=True)

    def _f2_merge_transaction_df_and_df_u_each_age_bin(self, unique_age_bin: str):
        """

        Parameters
        ----------
        unique_age_bin : str
            _description_
        """
        self.df_t_each_agebin = pd.merge(
            left=self.transaction_train,
            right=self.df_u_each_age_bin,
            on='customer_id_short',
            how='innor'
        )
        print(
            f'The shape of scope transaction for {unique_age_bin} is {self.df_t_each_agebin.shape}. \n')

    def _f3_create_ldbw_column(self):

        self.last_ts = self.df_t_each_agebin['t_dat'].max()
        tmp = self.df_t_each_agebin[['t_dat']].copy()
        # 曜日カラムを生成。dayofweek属性は、曜日のindex(月曜=0, 日曜=6)を返す。
        tmp['dow'] = tmp['t_dat'].dt.dayofweek

        # トランザクション発生週の最終日をlast_day_of_bought_weekとして保存?
        tmp['ldbw'] = tmp['t_dat'] - \
            pd.TimedeltaIndex(data=tmp['dow'] - 1, unit='D')
        tmp.loc[tmp['dow'] >= 2, 'ldbw'] = tmp.loc[tmp['dow'] >= 2, 'ldbw'] + \
            pd.TimedeltaIndex(
                np.ones(len(tmp.loc[tmp['dow'] >= 2])) * 7, unit='D')

        self.df_t_each_agebin['ldbw'] = tmp['ldbw'].values

    def _f4_add_weekly_sales(self):

        weekly_sales = self.df_t_each_agebin.drop('customer_id_short', axis=1).groupby(
            ['ldbw', 'article_id']).count().reset_index()
        weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})
        self.df_t_each_agebin = pd.merge(
            left=self.df_t_each_agebin,
            right=weekly_sales,
            on=['ldbw', 'article_id'],
            how='left'
        )

        weekly_sales = weekly_sales.reset_index().set_index('article_id')

        self.df_t_each_agebin = pd.merge(
            left=self.df_t_each_agebin,
            right=weekly_sales.loc[weekly_sales['ldbw'] == self.last_ts, ['count']],
            on='article_id',
            suffixes=("", "_targ")
        )

        self.df_t_each_agebin['count_targ'].fillna(0, inplace=True)
        del weekly_sales

    def create_reccomendation(self):

        # 各年齢Bin毎に繰り返し処理
        for unique_age_bin in self.list_UniBins:
            self._f1_extract_df_customer_each_age_bin(unique_age_bin)
            self._f2_merge_transaction_df_and_df_u_each_age_bin(unique_age_bin)

            pass

        self.df_sub = self.dataset.df_sub[['customer_id_short', 'customer_id']]

        # 最終的には3つのカラムにする.
        self.df_sub = self.df_sub[[
            'customer_id_short', 'customer_id', 'prediction']]

        return self.df_sub


if __name__ == '__main__':
    pass
