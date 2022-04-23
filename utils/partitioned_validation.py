from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
from multiprocessing.spawn import import_main_path
import os
from logging import lastResort
from utils.calculate_MAP12 import calculate_mapk, calculate_apk, mapk, apk
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

INPUT_DIR = 'input'
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


def divide_transaction_data_with_group(dataset: DataSet, divide_column: str) -> pd.DataFrame:
    pass


def partitioned_validation(val_df: pd.DataFrame, pred_df: pd.DataFrame, grouping: pd.Series, score: pd.DataFrame = 0, approach_name: str = "last_purchased_items", ignore: bool = False, figsize=(12, 6)) -> pd.DataFrame:
    """全ユーザのレコメンド結果を受け取り、グルーピング毎に予測精度を評価する関数。

    Parameters
    ----------
    val_df : pd.DataFrame
        レコメンドにおける実測値(各ユーザの指定された一週間に購入したarticle_id達)が格納されたデータフレーム
    pred_df : DataSet
        レコメンドにおける予測値(各ユーザの指定された一週間に購入したarticle_id達)が格納されたデータフレーム
    grouping : pd.Series
        各ユーザ毎のグルーピングを示すカテゴリ変数が格納されたpd.Series。
    score : pd.DataFrame, optional
        オフライン予測精度のスコアを格納していくDataFrame。 
        一度関数を実行する毎に、結果をレコードに追加していく。カラムはグルーピング。by default 0
    index : str, optional
        scoreのindex名,, by default -1
    ignore : bool, optional
        _description_, by default False
    figsize : tuple, optional
        _description_, by default (12, 6)

    Returns
    -------
    _type_
        _description_
    """

    # val_df["article_id"], dataset.df_sub["last_purchased_items"]からactual, predictedを抽出する.
    # レコードの順番をそろえたい...。
    val_df = val_df.sort_values(by='customer_id_short')
    pred_df = pred_df.sort_values(by='customer_id_short')
    print(val_df[['customer_id_short', 'article_id']].head())
    print(pred_df[['customer_id_short', f'prediction']].head())

    # Listで抽出
    actual: List[List[str]] = val_df['article_id'].apply(
        lambda s: [] if pd.isna(s) else s.split())
    predicted: List[List[str]] = pred_df['prediction'].apply(
        lambda s: [] if pd.isna(s) else s.split())

    k = 12

    # もしignore==Trueだったら、この関数は終了。
    if ignore:
        return

    # 各ユーザのAP@kを算出(後半のヒストグラム作成の為に)
    ap12 = mapk(actual, predicted, return_apks=True)
    # MAP@kを算出
    map12 = round(np.mean(ap12), 6)

    # isinstance()関数でオブジェクトのデータ型を判定
    # 本来はscoreに各Validation結果を格納していく？
    # scoreがDataFrameじゃなかったら、結果格納用のDataFrameをInitialize
    if isinstance(score, int):
        score = pd.DataFrame({g: []
                             for g in grouping.unique().tolist()})

    # もしindex引数が－１だったら...何の処理?
    if approach_name == -1:
        approach_name = score.shape[0]

    # 結果をDataFrameに保存
    score.loc[approach_name, "All"] = map12

    # グルーピング毎のValidation結果を作成
    for g in grouping.unique():
        map12 = round(mapk(actual[grouping == g], predicted[grouping == g]), 6)
        # score:DataFrameに結果を格納
        score.loc[approach_name, g] = map12
        print(map12)

    # レコメンドアイテム数は制限(12個×全ユーザ)の何%を占めてる？
    vc = pd.Series(predicted).apply(len).value_counts()
    # PercentageをFILLカラムに格納。
    score.loc[approach_name, "Fill"] = round(
        1 - sum(vc[k] * (12 - k) / 12 for k in (set(range(12)) & set(vc.index))) / len(actual), 3) * 100

    return score


def user_grouping_online_and_offline(dataset: DataSet) -> pd.DataFrame:
    """Datasetオブジェクトを受け取って、各ユーザの「オンライン販売かオフライン販売のどちらで多く購入する習慣があるか」でグルーピングする関数。

    Parameters
    ----------
    dataset : DataSet
        _description_

    Returns
    -------
    pd.DataFrame
        グルーピング結果を格納したDataFrame。
    """
    grouping_column: str = "sales_channel_id"

    # ユーザレコードの補完用にcustomer_id_dfを使う.
    alluser_df = dataset.cid

    if 'customer_id_short' in dataset.df.columns:
        # defaultでは、各ユーザが「オンライン販売かオフライン販売」のどちらで多く購入する週間があるかでグルーピングしてる。
        group: pd.DataFrame = dataset.df.groupby('customer_id_short')[
            grouping_column].mean().round().reset_index()
    # alluser_dfとグルーピングをマージする
    group = pd.merge(group, alluser_df, on='customer_id_short', how='right').rename(
        columns={grouping_column: f'group'})
    # 欠損値は1で埋める。１と２の違いって何？オンライン販売かオフライン販売？
    grouping_df = group[['customer_id_short', f'group']].fillna(1.0)

    return grouping_df


def user_grouping_age_bin(dataset: DataSet) -> pd.DataFrame:
    """Datasetオブジェクトを受け取って、
    各ユーザの年齢層でグルーピングする関数。

    Parameters
    ----------
    dataset : DataSet
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    # ユーザのメタデータを使用する
    df_customer = dataset.dfu
    # 年齢層毎にグルーピング
    listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
    df_customer['age_bins'] = pd.cut(df_customer['age'], listBin)
    # 年齢の欠損値の数を確認
    x = df_customer[df_customer['age_bins'].isnull()].shape[0]
    print(f'{x} customer_id do not have age information.\n')
    # =>とりあえず欠損値のままでOK？

    # 返値用のdfを生成
    df_customer.rename(columns={'age_bins': 'group'}, inplace=True)
    grouping_df = df_customer[['customer_id_short', 'group']]

    return grouping_df


def main():
    sample1_list = list(range(5))
    print(iter_to_str(iterable=sample1_list))


if __name__ == '__main__':
    main()
    datetime.time
