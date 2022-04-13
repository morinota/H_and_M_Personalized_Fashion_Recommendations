from sympy import Li
from calculate_MAP12 import calculate_mapk, calculate_apk

from dataset import DataSet
import seaborn as sns
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def _blend(reccomends: pd.Series, weights: List = [], k: int = 12) -> str:
    """_summary_
     各レコメンド手法のレコメンド結果を良い感じにブレンドする関数。apply()メソッド用。
     具体的には、各レコメンド手法への重み付け＋レコメンドの重複を元に、スコアを算出。
     スコアが高い12個のレコメンド商品をアンサンブル結果とする。
    Parameters
    ----------
    reccomends : pd.Series
        indexが各レコメンド手法。要素はある一ユーザへのレコメンド商品のid達が連結したstr.
        pd.DataFrame.apply(axis=1)で、各レコードに対して処理が適用される。
    weights : list, optional
        各レコメンド手法の重み付けのリスト。重みが大きい方が重要度が高い, by default []
    k : int, optional
        レコメンドアイテムの数, by default 12

    Returns
    -------
    str
        各レコメンド手法のレコメンド結果を、重み付けを元に良い感じにブレンドした結果の文字列。
    """

    # もし重み付けが設定されていなければ、
    if len(weights) == 0:
        # 重み一律でweightsをInitialize
        weights = [1] * len(reccomends)

    # 全レコメンド結果を格納するListをInitialize
    preds: List[str] = []

    # 各レコメンド手法毎に処理
    for i in range(len(weights)):
        # レコメンドアイテムのListを1つの文字列に変換してPredsに格納
        preds.append(reccomends[i].split())

    # 返値用のDictをInitialize
    res: Dict[str, float] = {}

    # 全レコメンドアイテムに対して、重み付けを考慮して合体させる。
    # 各レコメンド手法毎に処理
    for i in range(len(preds)):
        # もし重みが0より小さければ、そもそもレコメンド対象にいれず、次のレコメンド手法へ。
        if weights[i] < 0:
            continue
        # 各レコメンド手法のレコメンドアイテム達を個々に処理。
        for j, article_id in enumerate(preds[i]):

            # もし結果格納用のDictのkeyに、すでにアイテムが含まれていれば、
            if article_id in res:
                # 重み付けの分をスコアに加える。
                res[article_id] += (weights[i] / (j + 1))
            # 初見のアイテムの場合は、S
            else:
                # 結果格納用のDictに新規追加。
                res[article_id] = (weights[i] / (j + 1))

    # スコアが大きい順にarticle_idをソート！＝＞Listに。
    res_list: List[str]
    res_list = list(
        dict(sorted(res.items(), key=lambda item: -item[1])).keys())

    # スコアが大きいk個のarticle_idを切り出し、" "で繋いだ文字列にしてReturn
    return ' '.join(res_list[:k])


def _prune(pred: str, ok_set: Set[int], k: int = 12) -> str:
    """_summary_
    各レコメンド手法のレコメンド結果を良い感じにブレンドした後、良い感じに切り落とす関数。apply()メソッド用
    具体的には、ある任意の期間に誰にも全く購入されていないアイテムは除外する。
    Parameters
    ----------
    pred : str
        各ユーザへのレコメンドアイテム達。
    ok_set : Set[int]
        学習対象期間内(基本的には今シーズン)に一回でも誰かに購入されたアイテムのリスト
    k : int, optional
        レコメンドアイテムの数, by default 12

    Returns
    -------
    str
        各ユーザの最終的なレコメンドアイテム達。
    """
    # 処理しやすくする為に、一旦str=>Listに
    pred_list: List[str] = pred.split(" ")

    # 結果格納用のListをInitialize
    post: List[str] = []

    # 各レコメンドアイテムに対して繰り返し処理
    for article_id in pred_list:
        # もしarticle_idがok_setの中に含まれており、且つpostにまだ入って無ければ。
        if int(article_id) in ok_set and not article_id in post:
            # 追加
            post.append(article_id)

    # 再びList＝＞Strに戻してReturn
    return " ".join(post[:k])


def recommend_emsemble(predicted_kwargs:Dict[str, pd.DataFrame], weight_args:List, dataset: DataSet, val_week_id: int = 105)->pd.DataFrame:
    """_summary_

    Parameters
    ----------
    predicted_kwargs : Dict[str, pd.DataFrame]
        _description_
    weight_args : List
        _description_
    dataset : DataSet
        _description_
    val_week_id : int, optional
        _description_, by default 105

    Returns
    -------
    pd.DataFrame
        _description_
    """

    # ok_setを作る。(今シーズン一回も誰にも買われてないアイテムはレコメンドしない)
    df = dataset.df
    last_date = df.loc[df.week < val_week_id].t_dat.max()
    init_date = last_date - datetime.timedelta(days=11)
    sold_set = set(df.loc[(df.t_dat >= init_date) & (
        df.t_dat <= last_date)].article_id.tolist())

    submission_df: pd.DataFrame
    i = 0
    # DataFrameを合成
    for k, v in predicted_kwargs.items():
        if i = 0:
            submission_df = v
        else:
            submission_df.merge(
                right=predicted_kwargs[i], on='customer_id_short')
        # カラム名を変更
        submission_df.rename(columns={'predicted': k})
        i += 1

    # 各レコメンド結果を重み付け
    p_columns_list = predicted_kwargs.keys()
    submission_df['prediction'] = submission_df[p_columns_list].apply(
        _blend, weight_args, acis=1, k=32).apply(_prune, ok_set=sold_set)

    return submission_df[['customer_id_short', 'prediction']]