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

# 特徴量生成のベースとなるクラス


class UserFeatures(ABC):
    @abstractmethod
    def get(self) -> pd.DataFrame:
        """
        customer_id -> features
        """
        pass


class AggrFeatures(UserFeatures):
    """
    トランザクションログをベースに、ユーザ毎のNumericalデータの特徴量作成。
    Numericalデータとして使えそうなカラムはPriceと
    basic aggregation features(min, max, mean and etc...)
    """

    def __init__(self, transactions_df: pd.DataFrame):
        # ユーザ毎にトランザクションログをGroupby
        self.groupby_df = transactions_df.groupby(
            'customer_id_short', as_index=False)

    def get(self):
        # トランザクション価格に対して、たくさん特徴量を作成。
        output_df_price = (
            self.groupby_df['price']
            .agg({
                'mean_transaction_price': 'mean',  # 購入価格の平均
                # 'variance_transaction_price': lambda x: x.val(), # 購入価格の分散
                'std_transaction_price': lambda x: x.std(),  # 購入価格の標準偏差
                'max_transaction_price': 'max',  # 購入価格の最大値
                'min_transaction_price': 'min',  # 購入価格の最小値
                'median_transaction_price': 'median',  # 購入価格の中央値
                'sum_transaction_price': 'sum',  # 購入価格の合計値
                # maxとminの差
                'max_minus_min_transaction_price': lambda x: x.max()-x.min(),
                # maxとmeanの差
                'max_minus_mean_transaction_price': lambda x: x.max()-x.mean(),
                # minとmeanの差
                'mean_minus_min_transaction_price': lambda x: x.mean()-x.min(),
                # sum/mean = count (トランザクション回数)
                'count_transaction_price': lambda x: x.sum() / x.mean(),

                # 小数点以下だけ取り出した要素
                'mean_transaction_price_under_point': lambda x: math.modf(x.mean())[0],
                'mean_transaction_price_over_point': lambda x: math.modf(x.mean())[1],
                'max_transaction_price_under_point': lambda x: math.modf(x.max())[0],
                'max_transaction_price_over_point': lambda x: math.modf(x.max())[1],
                'min_transaction_price_under_point': lambda x: math.modf(x.min())[0],
                'min_transaction_price_over_point': lambda x: math.modf(x.min())[1],
                'median_transaction_price_under_point': lambda x: math.modf(x.median())[0],
                'median_transaction_price_over_point': lambda x: math.modf(x.median())[1],
                'sum_transaction_price_under_point': lambda x: math.modf(x.sum())[0],
                'sum_transaction_price_over_point': lambda x: math.modf(x.sum())[1],
            })
            .set_index('customer_id_short')
            .astype('float32')
        )

        # トランザクションのオンライン/オフラインに対して、特徴量を作成
        output_df_sales_channel_id = (
            self.groupby_df['sales_channel_id']
            .agg({
                'mean_sales_channel_id': 'mean',
                'median_sales_channel_id': 'median',
                'sum_sales_channel_id': 'sum'
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

        # 最終的なReturnは、ユニークユーザがレコードのDataFrame
        return output_df


class CountFeatures(UserFeatures):
    """
    トランザクションログをベースに、
    basic features connected with transactions
    """

    def __init__(self, transactions_df, topk=10):
        self.transactions_df = transactions_df
        self.topk = topk

    def get(self):
        # ユーザ毎にトランザクションログをGroupby
        grouped = self.transactions_df.groupby(
            'customer_id_short', as_index=False)

        a = (
            grouped
            .agg({
                # トランザクション回数
                'article_id': 'count',
                # 平均購入価格よりも高い金額で購入した回数
                'price': lambda x: sum(np.array(x) > x.mean()),
                # 2(オンラインで購入した回数)
                'sales_channel_id': lambda x: sum(x == 2),
            })
            .rename(columns={
                'article_id': 'n_transactions',
                'price': 'n_transactions_bigger_mean',
                'sales_channel_id': 'n_online_articles'
            })
            .set_index('customer_id_short')
            .astype('int8')
        )

        b = (
            grouped
            .agg({
                # 購入したアイテムの中の、ユニークアイテム数
                'article_id': 'nunique',
                # 1(オフライン販売)で購入した回数。
                'sales_channel_id': lambda x: sum(x == 1),
            })
            .rename(columns={
                'article_id': 'n_unique_articles',
                'sales_channel_id': 'n_store_articles',
            })
            .set_index('customer_id_short')
            .astype('int8')
        )

        topk_articles = self.transactions_df['article_id'].value_counts()[
            :self.topk].index
        c = (
            # 売上人気アイテムを購入した回数.
            grouped['article_id']
            .agg({
                f'top_article_{i}': lambda x: sum(x == k) for i, k in enumerate(topk_articles)
            }
            )
            .set_index('customer_id_short')
            .astype('int8')
        )

        output_df = a.merge(b, on=('customer_id_short')).merge(
            c, on=('customer_id_short'))
        return output_df


class CustomerFeatures(UserFeatures):
    """
    All columns from customers dataframe
    """

    def __init__(self, customers_df):
        self.customers_df = customers_df

    def _completion_isMissing_value(self):
        """欠損値を補完するメソッド
        """

        self.customers_df['FN'] = self.customers_df['FN'].fillna(
            0).astype('int8')
        self.customers_df['Active'] = self.customers_df['Active'].fillna(
            0).astype('int8')
        self.customers_df['club_member_status'] = self.customers_df['club_member_status'].fillna(
            'UNKNOWN')
        self.customers_df['age'] = self.customers_df['age'].fillna(
            self.customers_df['age'].mean()).astype('int8')
        self.customers_df['fashion_news_frequency'] = (
            self.customers_df['fashion_news_frequency']
            .replace('None', 'NONE')
            .replace(np.nan, 'NONE')
        )
        return self.customers_df

    def _create_isnull_column(self):
        """isnullカラムを生成するメソッド。
        """
        def __create_isnull_column(column_name: str):

            self.customers_df[f'{culumn_name}_is_null'] = (
                self.customers_df[column_name].isna()
            )
            # boolを0/1に変換
            self.customers_df[f'{culumn_name}_is_null'] = (
                self.customers_df[f'{culumn_name}_is_null'] * 1
            )
        columns_list = ['FN', 'Active', 'club_member_status',
                        'fashion_news_frequency', 'age']
        for column_name in columns_list:
            __create_isnull_column(column_name)

    def get(self):
        output = (
            self.customers_df[filter(
                lambda x: x != 'postal_code', self.customers_df.columns)]
            .set_index('customer_id_short')
        )
        return output


class ArticlesFeatures(UserFeatures):
    """
    トランザクションログとアイテムメタデータをベースに、ユーザ毎のNumericalデータの特徴量作成。
    returns article features: whether category appears in top categories
    """

    def __init__(self, transactions_df, articles, topk=10):
        # トランザクションログに、アイテムメタデータを付与したもの
        self.transactions_merged_df = transactions_df.merge(
            articles, on=('article_id'))
        # アイテムメタ
        self.articles = articles
        self.topk = topk

    def get(self):
        output_df = None

        # アイテムメタの各カラム毎に繰り返し処理
        for col in tqdm(self.articles.columns, desc='extracting features'):
            # もしカラム名に'name'が含まれている=カテゴリ変数ならば...
            if 'name' in col:
                if output_df is None:
                    output_df = self._aggregate_topk(
                        self.transactions_merged_df, col, self.topk)
                else:
                    intermediate_out = self._aggregate_topk(
                        self.transactions_merged_df, col, self.topk)
                    output_df = output_df.merge(
                        intermediate_out, on=('customer_id'))
        return output_df

    def _return_value_counts(self, df:pd.DataFrame, column_name:str, k:int)->List[int]:
        """カラムを指定して、データの値の頻度を計算し、上位アイテムk個の、出現頻度のリストを返す??

        Parameters
        ----------
        df : _type_
            _description_
        column_name : _type_
            _description_
        k : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # カラムを指定して、データの値の頻度を計算し、上位アイテムk個のみ抽出
        value_counts = df[column_name].value_counts()[:k].index
        value_counts = list(map(lambda x: x[1], value_counts))
        return value_counts

    def _aggregate_topk(self, merged_df:pd.DataFrame, column_name:str, k:int):
        # トランザクションログをユーザidでグルーピング(グループラベルをIndexにする)
        grouped_df_indx = merged_df.groupby('customer_id_short')
        # グループラベルをIndexにしないVer.
        grouped_df = merged_df.groupby('customer_id_short', as_index=False)

        # 各ユーザ毎に、対象カラムにおいて、高頻度カテゴリの情報を取得する。
        # ex.）ユーザAはズボン系を良く買う。とか？
        topk_values = self._return_value_counts(
            grouped_df_indx, column_name, k)

        # how many transactions appears in top category(column)
        # どれだけのトランザクションが、対象カラムにおける高頻度カテゴリに属しているか？
        n_top_k = (
            grouped_df[column_name]
            .agg({
                f'top_{column_name}_{i}': lambda x: sum(x == k) for i, k in enumerate(topk_values)
            })
            .set_index('customer_id')
            .astype('int16')
        )
        return n_top_k


class UserFeaturesCollector:
    """最終的に全てのユーザ特徴量をくっつける為のクラス。
    collect all features and aggregate them
    """
    @staticmethod
    def collect(features: Union[List[UserFeatures], List[str]], **kwargs) -> pd.DataFrame:
        output_df = None

        for feature in tqdm(features):
            if isinstance(feature, UserFeatures):
                feature_out = feature.get(**kwargs)
            if isinstance(feature, str):
                try:
                    feature_out = pd.read_csv(feature)
                except:
                    feature_out = pd.read_parquet(feature)

            if output_df is None:
                output_df = feature_out
            else:
                output_df = output_df.merge(feature_out, on=('customer_id'))
        return output_df


if __name__ == '__main__':
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data(c_id_short=True)

    # データをDataFrame型で読み込み
    df_transaction = dataset.df
    df_sub = dataset.df_sub  # 提出用のサンプル
    df_customers = dataset.dfu  # 各顧客の情報(メタデータ)
    df_articles = dataset.dfi  # 各商品の情報(メタデータ)

    #
    user_features = UserFeaturesCollector.collect([
        AggrFeatures(transactions_train.iloc[:10_000]),
        CountFeatures(transactions_train.iloc[:10_000], 3),
        CustomerFeatures(customers_df),
        ArticlesFeatures(transactions_train.iloc[:10_000], articles_df, 3),
    ])
