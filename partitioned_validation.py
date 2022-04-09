from multiprocessing.spawn import import_main_path
import os
from logging import lastResort
from calculate_MAP12 import calculate_mapk, calculate_apk
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from dataset import DataSet 
from useful_func import iter_to_str

INPUT_DIR = 'input'
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'




def partitioned_validation(actual:List[List], predicted: List[List], grouping: pd.Series, score: pd.DataFrame = 0, index: str = -1, ignore: bool = False, figsize=(12, 6)):
    """_summary_
    全ユーザのレコメンド結果を受け取り、グルーピング毎に予測精度を評価する関数。
    Parameters
    ----------
    actual : List[List]
        レコメンドにおける実測値。(各ユーザの指定された一週間に購入したarticle_id達)
    predicted : List[List]
        レコメンドにおける予測値。
    grouping : pd.Series
        各ユーザ毎のグルーピングを示すカテゴリ変数が格納されたpd.Series。
    score : pd.DataFrame, optional
        オフライン予測精度のスコアを格納していくDataFrame。 
        一度関数を実行する毎に、結果をレコードに追加していく。カラムはグルーピング。by default 0
    index : str, optional
        _description_, by default -1
    ignore : bool, optional
        _description_, by default False
    figsize : tuple, optional
        _description_, by default (12, 6)

    Returns
    -------
    _type_
        _description_
    """
    # '''
    # 実測値と予測値を受け取って、レコメンド精度を評価する関数。
    # parameters
    # ===================
    # # actual, predicted : list of lists
    # # grouping : submission.csvと同じ長さの、pd.Series。要素は各ユーザが、定義したどのグループに属しているかのカテゴリ変数?
    # # score : pandas DataFrame
    # index(str)：scoreのindex名
    # '''
    k = 12
    # もしignore==Trueだったら、この関数は終了。
    if ignore:
        return
    # 各ユーザのAP@kを算出(後半のヒストグラム作成の為に)
    apk_all_users = []
    for actual_items, predicted_items in zip(actual, predicted):
        apk_each_user = calculate_apk(actual_items, predicted_items, k)
        apk_all_users.append(apk_each_user)

    # MAP@kを算出
    mapk = np.mean(apk_all_users)
    mapk = round(mapk, 6)

    # isinstance()関数でオブジェクトのデータ型を判定
    if isinstance(score, int):
        # 本来はscoreに各Validation結果を格納していく？
        # scoreがDataFrameじゃなかったら、結果格納用のDataFrameをInitialize
        score = pd.DataFrame({g: []
                             for g in sorted(grouping.unique().tolist())})

    # もしindex引数が－１だったら...何の処理?
    if index == -1:
        index = score.shape[0]

    # 結果をDataFrameに保存
    score.loc[index, "All"] = mapk

    # MAP@kの結果を描画。(「各ユーザのAP@kの値」を集計して、ヒストグラムへ。なお縦軸は対数軸！)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    sns.histplot(data=apk_all_users, log_scale=(0, 10), bins=20)
    plt.title(f"MAP@12 : {mapk}")

    # グルーピング毎のValidation結果を作成
    for g in grouping.unique():
        map12 = round(mapk(actual[grouping == g], predicted[grouping == g]), 6)
        score.loc[index, g] = map12

    # グルーピング毎にMAP@Kを描画
    plt.subplot(1, 2, 2)
    score[[g for g in grouping.unique()[::-1]] + ['All']
          ].loc[index].plot.barh()
    plt.title(f"MAP@12 of Groups")
    vc = pd.Series(predicted).apply(len).value_counts()
    score.loc[index, "Fill"] = round(
        1 - sum(vc[k] * (12 - k) / 12 for k in (set(range(12)) & set(vc.index))) / len(actual), 3) * 100
    display(score)
    return score


def get_valid_oneweek_holdout_validation(dataset:DataSet, val_week_id: int = 104) -> pd.DataFrame:
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
        'customer_id')['article_id'].apply(iter_to_str).reset_index()
    # ->レコード：各ユーザ、カラム：customer_id, 1週間の購入アイテム達のstr　(submission.csvと同じ形式)　

    # 上記のval_dfは、検証用weekでtransactionを発生させたユーザのみ。それ以外のユーザのレコードを付け足す。
    alluser_df = dataset.cid

    # val_dfに検証用weekでtransactionを発生させていないユーザのレコードを付け足す。
    val_df = pd.merge(val_df, alluser_df, how="right", on='customer_id')

    # List[List]でも返値を作成しておく。
    actual = val_df['artcle_id'].apply(lambda s: [] if pd.isna(s) else s.split())

    return val_df, actual


def get_train_oneweek_holdout_validation(dataset:DataSet, val_week_id: int = 104, training_days: int = 31, how: str = "from_init_date_to_last_date") -> pd.DataFrame:

    # 学習用データを作成する
    transaction_df_train = pd.DataFrame()

    # 学習データ戦略1
    if how == "from_init_date_to_last_date":
        # "検証用の一週間"の前日の日付を取得
        mask = dataset.df["week"]< val_week_id
        last_date: datetime.datetime = dataset.df[mask]["t_dat"].max()
        # 学習用データのスタートの日付を取得
        init_date: datetime.datetime = last_date - \
            datetime.timedelta(days=training_days)
        # 学習用データを作成
        train_mask = (dataset.df["t_dat"] >= init_date) & (dataset.df["t_dat"] <= last_date)
        transaction_df_train: pd.DataFrame = dataset.df[train_mask]

    # 学習データ戦略2(昨年の同じシーズンのトランザクションを使う)
    if how == "use_same_season_in_past":
        pass

    return transaction_df_train


def all_process_partitioned_validation():
    pass

def user_grouping_online_and_offline(dataset:DataSet):
    grouping_column: str = "sales_channel_id"

    # ユーザレコードの補完用にcustomer_id_dfを使う.
    alluser_df = dataset.cid

    # defaultでは、各ユーザが「オンライン販売かオフライン販売」のどちらで多く購入する週間があるかでグルーピングしてる。
    group:pd.DataFrame = dataset.df.groupby('customer_id')[
        grouping_column].mean().round().reset_index()
    # alluser_dfとグルーピングをマージする
    group = pd.merge(group, alluser_df, on='customer_id', how='right').rename(
        columns={grouping_column: f'group_{grouping_column}'})
    # 欠損値は1で埋める。１と２の違いって何？オンライン販売かオフライン販売？
    grouping = group[f'group_{grouping_column}'].fillna(1.0, inplace=True)

    return grouping



def make_user_grouping(transaction_df, customer_df: pd.DataFrame, grouping_column: str = "sales_channel_id") -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    transaction_df : pd.DataFrame
        _description_
    customer_df : pd.DataFrame
        _description_
    grouping_column : str, optional
        _description_, by default "sales_channel_id"

    Returns
    -------
    pd.Series
        _description_
    """

    transaction_columns = ["sales_channel_id"]
    customer_columns = []
    article_columns = []

    # 補完用にsample_
    sub_df = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
    alluser_series = pd.DataFrame(sub_df["customer_id"].apply(
        lambda s: int(s[-16:], 16))).astype(str)

    grouping = pd.Series()

    if grouping_column in transaction_columns:
        # defaultでは、各ユーザが「オンライン販売かオフライン販売」のどちらで多く購入する週間があるかでグルーピングしてる。
        group = transaction_df.groupby('customer_id')[
            grouping_column].mean().round().reset_index()
        # submission用のデータとグルーピングをマージする
        group = pd.merge(group, alluser_series, on='customer_id', how='right').rename(
            columns={grouping_column: 'group'})
        # 欠損値は1で埋める。１と２の違いって何？オンライン販売かオフライン販売？
        grouping: pd.Series = group["group"].fillna(1.0)

    return grouping


def main():
    sample1_list = list(range(5))
    print(iter_to_str(iterable=sample1_list))

if __name__ == '__main__':
    main()
    datetime.time
