from datetime import timedelta
from sqlite3 import Timestamp
from time import time
from typing import Dict, List, Tuple
from flask import Config
import pandas as pd
from more_itertools import last
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm
from config import Config

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
        """対象Agebinユーザのトランザクションのみを取り出す

        Parameters
        ----------
        unique_age_bin : str
            _description_
        """
        self.df_t_each_agebin = pd.merge(
            left=self.transaction_train,
            right=self.df_u_each_age_bin,
            on='customer_id_short',
            how='inner'
        )
        print(
            f'The shape of scope transaction for {unique_age_bin} is {self.df_t_each_agebin.shape}. \n')

    def _f3_create_ldbw_column(self):
        """「トランザクションの最終日から何週間前か」を表現するを意味する"ldbw"(last_day_of_bought_week)カラムを作る
        """

        # トランザクションログの最終日を取得
        self.last_ts = self.df_t_each_agebin['t_dat'].max()
        # 曜日カラムを生成。dayofweek属性は、曜日のindex(月曜=0, 日曜=6)を返す。
        self.df_t_each_agebin['dow'] = self.df_t_each_agebin['t_dat'].dt.dayofweek
        # 最終日は何曜日？？=>1=火曜日
        dow_last_ts = self.last_ts.day_of_week

        # 'ldbw'カラムを生成。
        # (TimedeltaIndexは、timedeltaの各要素に適用するVer)
        # (もしt_datが火曜日の場合は、ldbw=t_dat)
        # (もしt_datが月曜日の場合は、ldbw=t_dat-(-1)=t_dat+1=火曜日)
        self.df_t_each_agebin['ldbw'] = (
            self.df_t_each_agebin['t_dat']
            - pd.TimedeltaIndex(data=self.df_t_each_agebin['dow'] - dow_last_ts, unit='D')
        )

        # t_datが水曜日以降のレコードの場合は、次の週(次の火曜日)としてldbwをカウントする
        self.df_t_each_agebin.loc[self.df_t_each_agebin['dow'] >= 2, 'ldbw'] = (
            self.df_t_each_agebin.loc[self.df_t_each_agebin['dow'] >= 2, 'ldbw']
            + timedelta(days=7)
        )

    def _f4_1_calculate_weekly_sales(self):
        # 各アイテムのWeeklySalesを取得
        self.weekly_sales = self.df_t_each_agebin.groupby(
            ['ldbw', 'article_id'])['t_dat'].count().reset_index()

        self.weekly_sales = self.weekly_sales.rename(
            columns={'t_dat': 'count'})

        # トランザクションログにweekly_salesをマージ
        self.df_t_each_agebin = pd.merge(
            left=self.df_t_each_agebin,
            right=self.weekly_sales,
            on=['ldbw', 'article_id'],
            how='left'
        )

    def _f4_2_calculate_count_targ(self):
        self.weekly_sales = self.weekly_sales.reset_index().set_index('article_id')

        # count_targカラムを生成
        self.df_t_each_agebin = pd.merge(
            left=self.df_t_each_agebin,
            right=self.weekly_sales.loc[self.weekly_sales['ldbw']
                                        == self.last_ts, ['count']],
            on='article_id',
            # 列名が重複している場合の処理(デフォルトは'_x', '_y')
            suffixes=("", "_targ")
        )

        # count_targカラムの欠損を埋める
        self.df_t_each_agebin['count_targ'].fillna(0, inplace=True)
        del self.weekly_sales

    def _f4_3_calculate_quotient(self):
        self.df_t_each_agebin['quotient'] = (
            self.df_t_each_agebin['count_targ']/self.df_t_each_agebin['count']
        )

    def _f5_create_general_pred(self):
        """quotientの各アイテム毎の合計値を算出し、上位k個をgeneral_predとする。
        """
        # 各アイテム毎のquotientの合計値を算出
        target_sales = self.df_t_each_agebin.drop(
            'customer_id_short', axis=1).groupby('article_id')['quotient'].sum()
        # quotientの合計値の大きい、上位12商品のarticle_id(dfのindexになってる)を取得
        self.general_pred = target_sales.nlargest(n=12).index.tolist()

        # article_idを提出用に整形
        self.general_pred = [str(article_id).zfill(10) for article_id in self.general_pred]
        print(f'general pred is {self.general_pred}')
        del target_sales

    def _f6_conduct_byfone_2(self):
        """同じアイテムを再度レコメンドする戦略
        """
        self.purchase_dict = {}
        df = self.df_t_each_agebin
        # Byfone戦略2つ目
        for i in tqdm(df.index):
            cust_id = df.at[i, 'customer_id_short']
            art_id = df.at[i, 'article_id']
            t_dat = df.at[i, 't_dat']

            if cust_id not in self.purchase_dict:
                self.purchase_dict[cust_id] = {}

            if art_id not in self.purchase_dict[cust_id]:
                self.purchase_dict[cust_id][art_id] = 0

            # x：ユーザAがアイテムBを購入した日からトランザクションログ最終日までの経過日数。
            x = max(1, (self.last_ts - t_dat).days)

            # y:cによって減衰する値
            # (xが小さい＝yが大きいと、顧客が短期間に同じ製品を購入することを意味する?)
            a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
            y = a / np.sqrt(x) + b * np.exp(-c*x) - d

            # value : y * quotient
            # (顧客が同じ商品を短期間購入し(=yが大きい)、且つその商品の成長率が高ければ(=quotientが大きい)、さらに購入すると予想される。)
            value = df.at[i, 'quotient'] * max(0, y)
            self.purchase_dict[cust_id][art_id] += value


        # tmp = self.df_t_each_agebin.copy()
        # # x：ユーザAがアイテムBを購入した日からトランザクションログ最終日までの経過日数。
        # tmp['x'] = ((self.last_ts - tmp['t_dat']) /
        #             np.timedelta64(1, 'D')).astype(int)
        # # yの計算の準備
        # tmp['dummy_1'] = 1
        # tmp['x'] = tmp[["x", "dummy_1"]].max(axis=1)
        # a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
        # # y:cによって減衰する値(xが小さい＝yが大きいと、顧客が短期間に同じ製品を購入することを意味する?)
        # tmp['y'] = a / np.sqrt(tmp['x']) + b * np.exp(-c*tmp['x']) - d
        # tmp['dummy_0'] = 0
        # tmp['y'] = tmp[["y", "dummy_0"]].max(axis=1)
        
        # # value : y * quotient
        # # (顧客が同じ商品を短期間購入し(=yが大きい)、且つその商品の成長率が高ければ(=quotientが大きい)、さらに購入すると予想される。)
        # tmp['value'] = tmp['quotient'] * tmp['y']

        # # ユーザA、アイテムBに関するvalueの合計値を取得
        # tmp = tmp.groupby(['customer_id_short', 'article_id']
        #                   ).agg({'value': 'sum'})
        # tmp = tmp.reset_index()
        # tmp = tmp.loc[tmp['value'] > 0]

        # # 各ユーザA、アイテムBに対して、「valueの合計値」の上位12個のみを候補として残す。
        # tmp['rank'] = tmp.groupby("customer_id_short")[
        #     "value"].rank("dense", ascending=False)
        # tmp = tmp.loc[tmp['rank'] <= 12]

        # # 各ユーザ毎に、valueが高いアイテムを取り出す作業。
        # self.purchase_df = tmp.sort_values(
        #     ['customer_id_short', 'value'], ascending=False).reset_index(drop=True)
        # self.purchase_df['prediction'] = (
        #     [str(self.purchase_df['article_id']).zfill(10)]
        # )
        # self.purchase_df = self.purchase_df.groupby(
        #     'customer_id_short').agg({'prediction': sum}).reset_index()
        # self.purchase_df['prediction'] = self.purchase_df['prediction'].str.strip()

    def _f7_conduct_recommendation(self, uniBin):
        sub = self.dataset.df_sub[['customer_id_short', 'customer_id']].copy()
        self.numCustomers = sub.shape[0]
        self.k:int = Config.num_recommend_item

        # 対象のage binのユーザのみ残す
        sub = pd.merge(
            left=sub, right=self.df_u_each_age_bin[[
                'customer_id_short', 'age']],
            on='customer_id_short', how='inner',
        )

        # 両者のレコメンド結果を結合
        pred_list = []
        for cust_id in tqdm(self.df_sub['customer_id_short']):
            # もしユーザidがpurchase_dictにあれば=トランザクションログに含まれていれば...
            if cust_id in self.purchase_dict:
                series = pd.Series(self.purchase_dict[cust_id])
                series = series[series > 0]
                # valueの多い上位12個のarticle_idをリストで取得
                l = series.nlargest(self.k).index.tolist()
                # もしレコメンド数が足りなければGeneralPred
                if len(l) < self.k:
                    l = l + self.general_pred[:(self.k-len(l))]
            # もしユーザidがpurchase_dictになければ=トランザクションログに含まれていなければ...
            else:
                l = self.general_pred

            # リストを文字列に変換
            pred_list.append(' '.join([str(x).zfill(10) for x in l]))

        # 提出用に整形
        sub['prediction'] = pred_list

        # sub = pd.merge(
        #     left=sub, right=self.purchase_df,
        #     on='customer_id_short', how='left',
        #     suffixes=('', '_ignored')
        # )
        # # レコメンドの不足分を補完
        # sub['prediction'] = sub['prediction'].fillna(self.general_pred)
        # sub['prediction'] = sub['prediction'] + ' ' + self.general_pred_str
        # # 両端（先頭、末尾）の半角スペース文字を削除
        # sub['prediction'] = sub['prediction'].str.strip(' ')
        # # 12個にする
        # sub['prediction'] = sub['prediction'][:131] # 一旦リストに

        # 最終的には2つのカラムにする。
        sub = sub[['customer_id_short', 'prediction']]
        # 一旦エクスポート
        sub.to_csv(f'submission_' + str(uniBin) + '.csv', index=False)
        print(f'Saved prediction for {uniBin}. The shape is {sub.shape}. \n')
        print('-'*50)

    def create_reccomendation(self):

        # 各年齢Bin毎に繰り返し処理
        for unique_age_bin in self.list_UniBins:
            self._f1_extract_df_customer_each_age_bin(unique_age_bin)
            self._f2_merge_transaction_df_and_df_u_each_age_bin(unique_age_bin)
            self._f3_create_ldbw_column()
            self._f4_1_calculate_weekly_sales()
            self._f4_2_calculate_count_targ()
            self._f4_3_calculate_quotient()
            self._f5_create_general_pred()
            self._f6_conduct_byfone_2()
            self._f7_conduct_recommendation(unique_age_bin)

        # 各年齢bin毎の結果を結合
        self.df_sub = self.dataset.df_sub[[
            'customer_id_short', 'customer_id']].copy()

        for i, unique_age_bin in enumerate(self.list_UniBins):
            df_temp = pd.read_csv(
                f'submission_' + str(unique_age_bin) + '.csv')

            self.df_sub = pd.merge(self.df_sub, df_temp, how='left',
                                   on='customer_id_short')

        # もし欠損のあるユーザがいれば、''を埋める
        self.df_sub['prediction'].fillna(value='', inplace=True)

        # エラーメッセージ(もしレコメンド結果の長さが違ったら)
        assert self.df_sub.shape[
            0] == self.numCustomers, f'The number of dfSub rows is not correct. {self.df_sub.shape[0]} vs {self.numCustomers}.'

        # 最終的には3つのカラムにする.
        self.df_sub = self.df_sub[[
            'customer_id_short', 'customer_id', 'prediction']]

        return self.df_sub


if __name__ == '__main__':
    pass
