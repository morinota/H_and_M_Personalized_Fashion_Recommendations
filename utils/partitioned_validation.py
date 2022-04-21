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

def divide_transaction_data_with_group(dataset:DataSet, divide_column:str)->pd.DataFrame:
    pass


def partitioned_validation(val_df: pd.DataFrame, pred_df: pd.DataFrame, grouping: pd.Series, score: pd.DataFrame = 0, approach_name: str = "last_purchased_items", ignore: bool = False, figsize=(12, 6)):
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
    # val_df = val_df.sort_values(by='customer_id_short')
    # pred_df = pred_df.sort_values(by='customer_id_short')
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
                             for g in sorted(grouping.unique().tolist())})

    # もしindex引数が－１だったら...何の処理?
    if approach_name == -1:
        approach_name = score.shape[0]

    # 結果をDataFrameに保存
    score.loc[approach_name, "All"] = map12

    # MAP@kの結果を描画。(「各ユーザのAP@kの値」を集計して、ヒストグラムへ。なお縦軸は対数軸！)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    sns.histplot(data=ap12, log_scale=(0, 10), bins=20)
    plt.title(f"MAP@12 : {map12}")

    # グルーピング毎のValidation結果を作成
    for g in grouping.unique():
        map12 = round(mapk(actual[grouping == g], predicted[grouping == g]), 6)
        # score:DataFrameに結果を格納
        score.loc[approach_name, g] = map12
        print(map12)

    # バープロットの描画
    plt.subplot(1, 2, 2)
    score[[g for g in grouping.unique()[::-1]] + ['All']
          ].loc[approach_name].plot.barh()
    plt.title(f"MAP@12 of Groups")

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