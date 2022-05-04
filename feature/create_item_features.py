from re import S
from textwrap import fill
import pandas as pd
import numpy as np
from regex import B
from pytest import Item
from tqdm import tqdm
import math
from my_class.dataset import DataSet
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Union
from datetime import timedelta

import os

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

OBJECT_COLUMNS = ['article_id', 'prod_name', 'product_type_name', 'product_group_name',
                  'graphical_appearance_name', 'colour_group_name',
                  'perceived_colour_value_name', 'perceived_colour_master_name',
                  'department_name', 'index_code', 'index_name', 'index_group_name',
                  'section_name', 'garment_group_name', 'detail_desc'
                  ]

ITEM_CATEGORICAL_COLUMNS = ['article_id',
                            # 'prod_name',
                            'product_type_name', 'product_group_name',
                            'graphical_appearance_name', 'colour_group_name',
                            'perceived_colour_value_name', 'perceived_colour_master_name',
                            'department_name', 'index_name', 'index_group_name',
                            'section_name', 'garment_group_name'
                            ]


# 特徴量生成のベースとなるクラス
class ItemFeatures(ABC):
    @abstractmethod
    def get(self) -> pd.DataFrame:
        """
        customer_id -> features
        """
        pass


class SalesLagFeatures(ItemFeatures):
    """各アイテム(or各アイテムサブカテゴリ)毎の時系列の売上個数のラグを生成する関数
    """

    def __init__(self, dataset: DataSet, transaction_df: pd.DataFrame) -> None:
        self.transaction_df = transaction_df
        self.dataset = dataset
        # article_idのデータ型を統一しておく
        self.transaction_df['article_id'] = self.transaction_df['article_id'].astype(
            'int')
        self.dataset.dfi['article_id'] = self.dataset.dfi['article_id'].astype(
            'int')

    def get(self) -> pd.DataFrame:

        print(self.transaction_df.columns)

        self.item_feature = pd.DataFrame()
        self._get_sales_time_series_each_item_subcategory()
        self._create_lag_feature()
        self._create_rolling_window_features()
        self._create_expanding_window_features()
        self._export_each_timeseries_features()

        return self.item_feature

    def _get_sales_time_series_each_item_subcategory(self) -> Dict:
        """各アイテム毎(or各アイテムサブカテゴリ)の時系列の売上個数のDataFrameを作る。
        (ラグ特徴量を作る為の準備)
        イメージ：レコードが各アイテム(or各アイテムサブカテゴリ)、
        各カラムが各週の売上個数を示すDataFrame
        """
        # トランザクションログに、アイテムメタデータとユーザメタデータを付与する
        self.transaction_df = pd.merge(
            left=self.transaction_df,
            right=self.dataset.dfi,
            on='article_id',
            how='left'
        )

        # まず日付のカラムから日・月・年のカラムを生成
        self.transaction_df['t_day'] = self.transaction_df['t_dat'].dt.day
        self.transaction_df['t_month'] = self.transaction_df['t_dat'].dt.month
        self.transaction_df['t_year'] = self.transaction_df['t_dat'].dt.year

        self.time_series_sales_count_dict = {}
        print(self.transaction_df.columns)
        # 各アイテム(or各アイテムサブカテゴリ)毎に繰り返し処理
        for target_column in ITEM_CATEGORICAL_COLUMNS:
            print(target_column)
            df_sales_timeseries = self.transaction_df.groupby(
                by=[target_column, pd.Grouper(
                    key='t_dat', freq="W")]  # type: ignore
            )['customer_id_short'].count()
            # unstacking
            df_sales_timeseries = df_sales_timeseries.unstack(fill_value=0)
            # dictに保存
            self.time_series_sales_count_dict[target_column] = df_sales_timeseries

        return self.time_series_sales_count_dict

    def _create_lag_feature(self):
        self.time_series_lag_sales_count_dict = {}
        # 各アイテム(or各アイテムサブカテゴリ)毎に繰り返し処理
        for target_column in ITEM_CATEGORICAL_COLUMNS:
            print(f'create lag features of {target_column}')
            df_sample: pd.DataFrame = self.time_series_sales_count_dict[target_column]

            # ラグ特徴量の生成
            lag1 = df_sample.shift(1, axis=1)
            lag2 = df_sample.shift(2, axis=1)

            # 結合用にstacking
            lag1 = lag1.stack().reset_index().rename(
                columns={0: f'lag1_salescount_{target_column}'})
            lag2 = lag2.stack().reset_index().rename(
                columns={0: f'lag2_salescount_{target_column}'})
            # 結合
            lag_item_feature = pd.merge(
                left=lag1, right=lag2, on=[target_column, 't_dat'], how='left'
            )
            # dictに格納
            self.time_series_lag_sales_count_dict[target_column] = lag_item_feature

            del lag1, lag2, lag_item_feature

    def _create_rolling_window_features(self):
        self.time_series_rolling_sales_count_dict = {}
        # 各アイテム(or各アイテムサブカテゴリ)毎に繰り返し処理
        for target_column in ITEM_CATEGORICAL_COLUMNS:
            print(f'create rolling window features of {target_column}')
            df_sample: pd.DataFrame = self.time_series_sales_count_dict[target_column]

            # Rolling特徴量の生成
            roll_mean_5 = df_sample.shift(
                1, axis=1).rolling(window=5, axis=1).mean()
            roll_mean_10 = df_sample.shift(
                1, axis=1).rolling(window=10, axis=1).mean()
            roll_var_5 = df_sample.shift(
                1, axis=1).rolling(window=5, axis=1).var()
            roll_var_10 = df_sample.shift(
                1, axis=1).rolling(window=10, axis=1).var()

            # 結合用にstacking
            roll_mean_5 = roll_mean_5.stack().reset_index().rename(
                columns={0: f'rollmean_5week_salescount_{target_column}'})
            roll_mean_10 = roll_mean_10.stack().reset_index().rename(
                columns={0: f'rollmean_10week_salescount_{target_column}'})
            roll_var_5 = roll_var_5.stack().reset_index().rename(
                columns={0: f'rollvar_5week_salescount_{target_column}'})
            roll_var_10 = roll_var_10.stack().reset_index().rename(
                columns={0: f'rollvar_10week_salescount_{target_column}'})

            # 結合
            rolling_item_feature = pd.DataFrame()
            for i, df_feature in enumerate([roll_mean_5, roll_mean_10, roll_var_5, roll_var_10]):
                if i == 0:
                    rolling_item_feature = df_feature
                else:
                    rolling_item_feature = pd.merge(
                        left=rolling_item_feature, right=df_feature,
                        on=[target_column, 't_dat'], how='left'
                    )
            # dictに格納
            self.time_series_rolling_sales_count_dict[target_column] = rolling_item_feature

            del roll_mean_5, roll_mean_10, roll_var_5, roll_var_10, rolling_item_feature

    def _create_expanding_window_features(self):
        self.time_series_expanding_sales_count_dict = {}
        # 各アイテム(or各アイテムサブカテゴリ)毎に繰り返し処理
        for target_column in ITEM_CATEGORICAL_COLUMNS:
            print(f'create expanding window features of {target_column}')
            df_sample: pd.DataFrame = self.time_series_sales_count_dict[target_column]

            # Expanding特徴量の生成
            expanding_mean = df_sample.shift(
                1, axis=1).expanding(axis=1).mean()
            expanding_var = df_sample.shift(
                1, axis=1).expanding(axis=1).var()

            # 結合用にstacking
            expanding_mean = expanding_mean.stack().reset_index().rename(
                columns={0: f'expanding_mean_salescount_{target_column}'})
            expanding_var = expanding_var.stack().reset_index().rename(
                columns={0: f'expanding_var_salescount_{target_column}'})

            # 結合
            expanding_item_feature = pd.DataFrame()
            for i, df_feature in enumerate([expanding_mean, expanding_var]):
                if i == 0:
                    expanding_item_feature = df_feature
                else:
                    expanding_item_feature = pd.merge(
                        left=expanding_item_feature, right=df_feature,
                        on=[target_column, 't_dat'], how='left'
                    )
            # dictに格納
            self.time_series_expanding_sales_count_dict[target_column] = expanding_item_feature

            del expanding_mean, expanding_var, expanding_item_feature

    def _export_each_timeseries_features(self):

        for target_column in ITEM_CATEGORICAL_COLUMNS:
            lag_features = self.time_series_lag_sales_count_dict[target_column]
            rolling_features = self.time_series_rolling_sales_count_dict[target_column]
            expanding_features = self.time_series_expanding_sales_count_dict[target_column]

            # 結合
            time_series_item_features = pd.DataFrame()
            for i, df_feature in enumerate([lag_features, rolling_features, expanding_features]):
                if i == 0:
                    time_series_item_features = df_feature
                else:
                    time_series_item_features = pd.merge(
                        left=time_series_item_features, right=df_feature,
                        on=[target_column, 't_dat'], how='left'
                    )

            # とりあえずアイテムの各サブカテゴリ毎の時系列特徴量をexportしておく
            file_path = os.path.join(
                DRIVE_DIR, f'feature/time_series_item_feature_{target_column}.csv')
            time_series_item_features.to_csv(file_path, index=False)
            pass


class NumericalFeature(ItemFeatures):
    def __init__(self, dataset: DataSet, transaction_df: pd.DataFrame) -> None:
        self.transaction_df = transaction_df
        self.dataset = dataset
        # article_idのデータ型を統一しておく
        self.transaction_df['article_id'] = self.transaction_df['article_id'].astype(
            'int')
        self.dataset.dfi['article_id'] = self.dataset.dfi['article_id'].astype(
            'int')

        # トランザクションログに、アイテムメタデータとユーザメタデータを付与する
        self.transaction_df = pd.merge(
            left=self.transaction_df,
            right=self.dataset.dfi,
            on='article_id',
            how='left'
        )

    def get(self) -> pd.DataFrame:

        numerical_features = self._get_item_feature_numerical()

        return numerical_features

    def _get_item_feature_numerical(self):
        """アイテムに関するNumerical特徴量を生成する関数
        Returns
        -------
        _type_
            _description_
        """
        self.groupby = self.transaction_df.groupby(by='article_id')

        # アイテム毎のトランザクション価格に対して、たくさん特徴量を生成
        output_df_price = (
            self.groupby['price']
            .agg(
                mean_item_price='mean',
                std_item_price=lambda x: x.std()
                # dictでaggとrenameを同時に行うのが非推奨らしい.
                #     {
                #     'mean_item_price': 'mean',
                #     'std_item_price': lambda x: x.std(),
                #     'max_item_price': 'max',
                #     'min_item_price': 'min',
                #     'median_item_price': 'median',
                #     'sum_item_price': 'sum',
                #     # maxとminの差
                #     'max_minus_min_item_price': lambda x: x.max()-x.min(),
                #     # maxとmeanの差
                #     'max_minus_mean_item_price': lambda x: x.max()-x.mean(),
                #     # minとmeanの差
                #     'mean_minus_min_item_price': lambda x: x.mean()-x.min(),
                #     # sum/mean = count (アイテムのトランザクション回数)
                #     'count_item_price': lambda x: x.sum() / x.mean(),
                #     # 小数点以下だけ取り出した要素
                #     'mean_item_price_under_point': lambda x: math.modf(x.mean())[0],
                #     'mean_item_price_over_point': lambda x: math.modf(x.mean())[1],
                #     'max_item_price_under_point': lambda x: math.modf(x.max())[0],
                #     'max_item_price_over_point': lambda x: math.modf(x.max())[1],
                #     'min_item_price_under_point': lambda x: math.modf(x.min())[0],
                #     'min_item_price_over_point': lambda x: math.modf(x.min())[1],
                #     'median_item_price_under_point': lambda x: math.modf(x.median())[0],
                #     'median_item_price_over_point': lambda x: math.modf(x.median())[1],
                #     'sum_item_price_under_point': lambda x: math.modf(x.sum())[0],
                #     'sum_item_price_over_point': lambda x: math.modf(x.sum())[1],
                # }
            )
            .set_index('article_id')
            .astype('float32')  # numerical 特徴量は全てfloatに
        )

        # トランザクションのオンライン/オフラインに対して、特徴量を作成
        output_df_sales_channel_id = (
            self.groupby['sales_channel_id']
            .agg({
                'item_mean_offline_or_online': 'mean',
                'item_median_offline_or_online': 'median',
                'item_sum_offline_or_online': 'sum'
            })
            .set_index('article_id')
            .astype('float32')  # numerical 特徴量は全てfloatに
        )

        # 横に結合
        self.output_df = pd.merge(
            output_df_price,
            output_df_sales_channel_id,
            how='left',
            left_index=True, right_index=True
        )

        return self.output_df


class CategoricalFeature(ItemFeatures):
    def __init__(self, dataset: DataSet, transaction_df: pd.DataFrame) -> None:
        self.transaction_df = transaction_df
        self.dataset = dataset
        # article_idのデータ型を統一しておく
        self.transaction_df['article_id'] = self.transaction_df['article_id'].astype(
            'int')
        self.dataset.dfi['article_id'] = self.dataset.dfi['article_id'].astype(
            'int')

        # トランザクションログに、アイテムメタデータとユーザメタデータを付与する
        self.transaction_df = pd.merge(
            left=self.transaction_df,
            right=self.dataset.dfi,
            on='article_id',
            how='left'
        )

    def get_item_feature_categorical(self):

        dfi_categorical = self.dfi[ITEM_CATEGORICAL_COLUMNS]

        def _extract_recent_transaction():
            """最近(直近三週間)のトランザクションのみを抽出。
            """
            # 最近(直近三週間)のトランザクションのみを抽出。
            last_date = self.transaction_with_itemmeta['t_dat'].max()
            init_date = pd.to_datetime(last_date) - timedelta(days=21)

            df_recent_t = self.transaction_with_itemmeta.loc[
                self.transaction_with_itemmeta['t_dat'] >= init_date
            ]
            return df_recent_t

        self.transaction_with_itemmeta_recent = _extract_recent_transaction()

        # これらの各カラムについて、人気ランキング的なモノを取得していく？
        def _create_popular_ranking_recently_with_each_category():
            df_popular_target = feature_eng.create_popular_ranking_in_transaction_log(
                df_transaction=self.transaction_with_itemmeta_recent,
                target_column='product_type_name',
            )

        self.dfi_popular = _create_popular_ranking_recently_with_each_category()

        self.item_feature_categorical = pd.DataFrame()
        return self.item_feature_categorical


class TargetEncodingFeature(ItemFeatures):
    def __init__(self, dataset: DataSet, transaction_df: pd.DataFrame) -> None:
        self.transaction_df = transaction_df
        self.dataset = dataset
        # article_idのデータ型を統一しておく
        self.transaction_df['article_id'] = self.transaction_df['article_id'].astype(
            'int')
        self.dataset.dfi['article_id'] = self.dataset.dfi['article_id'].astype(
            'int')

    def get(self) -> pd.DataFrame:

        self.item_feature = pd.DataFrame()

        return self.item_feature


def create_items_features():

    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data(c_id_short=False)

    # データをDataFrame型で読み込み
    df_transaction = dataset.df

    # item_lag_features
    # sales_lag_features = SalesLagFeatures(
    #     dataset=dataset, transaction_df=dataset.df)
    # print('create sales lag feature instance')
    # sales_lag_features.get()

    # numerical features
    numerical_item_feature = NumericalFeature(
        dataset=dataset, transaction_df=dataset.df
    )
    item_numerical_feature = numerical_item_feature.get()

    # エクスポート
    feature_dir = os.path.join(DRIVE_DIR, 'input')
    item_numerical_feature.to_parquet(os.path.join(
        feature_dir, 'item_numerical_features_my_fullT.parquet'), index=False)
