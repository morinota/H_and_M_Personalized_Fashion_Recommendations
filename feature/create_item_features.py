import pandas as pd
import numpy as np
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
from utils_feature_eng.datetime_feature import DatetimeFeature


OBJECT_COLUMNS = ['article_id', 'prod_name', 'product_type_name', 'product_group_name',
                  'graphical_appearance_name', 'colour_group_name',
                  'perceived_colour_value_name', 'perceived_colour_master_name',
                  'department_name', 'index_code', 'index_name', 'index_group_name',
                  'section_name', 'garment_group_name', 'detail_desc'
                  ]

ITEM_CATEGORICAL_COLUMNS = ['prod_name', 'product_type_name', 'product_group_name',
                  'graphical_appearance_name', 'colour_group_name',
                  'perceived_colour_value_name', 'perceived_colour_master_name',
                  'department_name', 'index_code', 'index_name', 'index_group_name',
                  'section_name', 'garment_group_name'
                  ]

# 特徴量生成のベースとなるクラス


class ItemFeatures():
    def __init__(self, dataset: DataSet, transaction_train: pd.DataFrame) -> None:
        self.dfi = dataset.dfi
        self.dfu = dataset.dfu
        self.transaction_train = transaction_train
        # article_idのデータ型を統一しておく
        self.transaction_train['article_id'] = self.transaction_train['article_id'].astype(
            'int')
        self.dfi['article_id'] = self.dfi['article_id'].astype('int')

    def get(self) -> pd.DataFrame:

        self.item_feature = pd.DataFrame()

        return self.item_feature

    def get_item_feature_numerical(self):

        def _merge_transaction_with_item_meta():
            """# トランザクションログに対して、アイテムメタをマージ。
            """
            transaction_more = pd.merge(
                self.transaction_train,
                self.dfi, on='article_id'
            )
            return transaction_more

        self.transaction_with_itemmeta = _merge_transaction_with_item_meta()

        def _grouping_transaction_each_item():
            # トランザクションログを、各アイテム毎にグルーピング
            grouped = self.transaction_with_itemmeta.groupby('article_id')

            return grouped

        self.groupby_df = _grouping_transaction_each_item()

        def _create_many_numerical_feature():
            # アイテム毎のトランザクション価格に対して、たくさん特徴量を生成
            output_df_price = (
                self.groupby_df['price']
                .agg({
                    'mean_item_price': 'mean',
                    'std_item_price': lambda x: x.std(),
                    'max_item_price': 'max',
                    'min_item_price': 'min',
                    'median_item_price': 'median',
                    'sum_item_price': 'sum',
                    # maxとminの差
                    'max_minus_min_item_price': lambda x: x.max()-x.min(),
                    # maxとmeanの差
                    'max_minus_mean_item_price': lambda x: x.max()-x.mean(),
                    # minとmeanの差
                    'mean_minus_min_item_price': lambda x: x.mean()-x.min(),
                    # sum/mean = count (アイテムのトランザクション回数)
                    'count_item_price': lambda x: x.sum() / x.mean(),
                    # 小数点以下だけ取り出した要素
                    'mean_item_price_under_point': lambda x: math.modf(x.mean())[0],
                    'mean_item_price_over_point': lambda x: math.modf(x.mean())[1],
                    'max_item_price_under_point': lambda x: math.modf(x.max())[0],
                    'max_item_price_over_point': lambda x: math.modf(x.max())[1],
                    'min_item_price_under_point': lambda x: math.modf(x.min())[0],
                    'min_item_price_over_point': lambda x: math.modf(x.min())[1],
                    'median_item_price_under_point': lambda x: math.modf(x.median())[0],
                    'median_item_price_over_point': lambda x: math.modf(x.median())[1],
                    'sum_item_price_under_point': lambda x: math.modf(x.sum())[0],
                    'sum_item_price_over_point': lambda x: math.modf(x.sum())[1],
                })
                .set_index('article_id')
                .astype('float32')
            )

            # トランザクションのオンライン/オフラインに対して、特徴量を作成
            output_df_sales_channel_id = (
                self.groupby_df['sales_channel_id']
                .agg({
                    'item_mean_offline_or_online': 'mean',
                    'item_median_offline_or_online': 'median',
                    'item_sum_offline_or_online': 'sum'
                })
                .set_index('customer_id_short')
                .astype('float32')
            )

            # 横に結合
            output_df = pd.merge(
                output_df_price,
                output_df_sales_channel_id,
                how='left',
                left_index=True, right_index=True
            )

            return output_df

        self.item_feature_numerical_from_transaction = _create_many_numerical_feature()

        return self.item_feature_numerical_from_transaction


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


if __name__ == '__main__':
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data(c_id_short=False)

    # データをDataFrame型で読み込み
    df_transaction = dataset.df
    df_sub = dataset.df_sub  # 提出用のサンプル
    df_customers = dataset.dfu  # 各顧客の情報(メタデータ)
    df_articles = dataset.dfi  # 各商品の情報(メタデータ)

    item_feature_class = ItemFeatures(dataset=dataset, transaction_train=df_transaction)
    item_feature = item_feature_class.get_item_feature_numerical()



    