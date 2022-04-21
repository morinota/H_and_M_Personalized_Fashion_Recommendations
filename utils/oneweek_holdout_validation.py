from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
from multiprocessing.spawn import import_main_path
import os
from logging import lastResort
from utils.calculate_MAP12 import calculate_mapk, calculate_apk
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def get_valid_oneweek_holdout_validation(dataset: DataSet, val_week_id: int = 104) -> pd.DataFrame:
    """トランザクションデータと検証用weekのidを受け取って、oneweek_holdout_validationの為の検証用データ(レコメンドの答え側)を作成する関数

    Parameters
    ----------
    dataset:Dataset

    val_week_id : int, optional
        2年分のトランザクションデータのうち、検証用weekに設定したいweekカラムの値, by default 104(2年の最終週)

    Returns
    -------
    pd.DataFrame
        oneweek_holdout_validationの為の検証用データ(レコメンドの答え側)。
        レコード：各ユーザ、カラム：customer_id, 1週間の購入アイテム達のstr　(submission.csvと同じ形式)のDataFrame
    """

    # 元々のtransaction_dfから、検証用weekのtransactionデータのみを抽出
    val_mask = (dataset.df['week'] == val_week_id)
    transaction_df_val = dataset.df[val_mask]

    # 検証用weekのtransactionデータから、検証用データ(レコメンドの答え側)を作成する。
    val_df: pd.DataFrame
    val_df = transaction_df_val.groupby(
        'customer_id_short')['article_id'].apply(iter_to_str).reset_index()
    # ->レコード：各ユーザ、カラム：customer_id, 1週間の購入アイテム達のstr　(submission.csvと同じ形式)　

    # 上記のval_dfは、検証用weekでtransactionを発生させたユーザのみ。それ以外のユーザのレコードを付け足す。
    alluser_df = dataset.cid

    # val_dfに検証用weekでtransactionを発生させていないユーザのレコードを付け足す。
    val_df = pd.merge(val_df, alluser_df, how="right", on='customer_id_short')

    # List[List]でも返値を作成しておく?
    actual = val_df['article_id'].apply(
        lambda s: [] if pd.isna(s) else s.split())

    return val_df


def get_train_oneweek_holdout_validation(dataset: DataSet, val_week_id: int = 104, training_days: int = 31, how: str = "from_init_date_to_last_date") -> pd.DataFrame:

    # 学習用データを作成する
    transaction_df_train = pd.DataFrame()

    # 学習データ戦略1
    if how == "from_init_date_to_last_date":
        # "検証用の一週間"の前日の日付を取得
        mask = dataset.df["week"] < val_week_id
        last_date: datetime.datetime = dataset.df[mask]["t_dat"].max()
        # 学習用データのスタートの日付を取得
        init_date: datetime.datetime = last_date - \
            datetime.timedelta(days=training_days)
        # 学習用データを作成
        train_mask = (dataset.df["t_dat"] >= init_date) & (
            dataset.df["t_dat"] <= last_date)
        transaction_df_train: pd.DataFrame = dataset.df[train_mask]

    # 学習データ戦略2(昨年の同じシーズンのトランザクションを使う)
    # ex) 2020年の8～9月のトランザクション + 2019年の同じ時期のトランザクション
    if how == "use_same_season_in_past":
        # "検証用の一週間"の前日の日付を取得
        mask = dataset.df["week"] < val_week_id
        last_date: datetime.datetime = dataset.df[mask]["t_dat"].max()
        # 学習用データ(2020年)のスタートの日付を取得
        init_date: datetime.datetime = last_date - \
            datetime.timedelta(days=training_days)
        # 学習用データ(2019年)のスタートとラストの日付を取得
        last_date_2019 = last_date - datetime.timedelta(days=365)
        init_date_2019 = init_date - datetime.timedelta(days=365)




    return transaction_df_train
