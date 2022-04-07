
from calculate_MAP12 import calculate_mapk, calculate_apk
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def _iter_to_str(iterable: List[str]) -> str:
    '''
    article_idの先頭に0を追加し、各article_idを半角スペースで区切られた文字列を返す関数
    (submitのcsvファイル様式にあわせる為に必要)

    parameters
    ===========
    iterable(ex. List)：イテラブルオブジェクト。ex. 各ユーザへのレコメンド商品のリスト。

    return
    ===========
    iterable_str(str)：イテラブルオブジェクトの各要素を" "で繋いで文字列型にしたもの。
    '''
    # Listの各要素の先頭に"0"を追加する
    iterable = map(lambda x: str(0) + str(x), iterable)
    # リストの要素を半角スペースで繋いで、文字列に。
    iterable_str = " ".join(iterable)
    return iterable_str


def _blend(reccomends: pd.Series, weights: List = [], k: int = 12) -> str:
    """_summary_
     各レコメンド手法のレコメンド結果を良い感じにブレンドする関数。apply()メソッド用
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


def _prune(pred: str, ok_set:Set[int], k:int=12)->str:
    """_summary_
    各レコメンド手法のレコメンド結果を良い感じにブレンドした後、良い感じに切り落とす関数。apply()メソッド用

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


def partitioned_validation(actual, predicted: List[List], grouping: pd.Series, score:pd.DataFrame=0, index:str=-1, ignore:bool=False, figsize=(12, 6)):
    """_summary_

    Parameters
    ----------
    actual : _type_
        _description_
    predicted : List[List]
        _description_
    grouping : pd.Series
        _description_
    score : pd.DataFrame, optional
        _description_, by default 0
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


def oneweek_holdout_validation():
    pass


def all_process_partitioned_validation():
    pass


def main():
    sample1_list = list(range(5))
    print(_iter_to_str(iterable=sample1_list))


if __name__ == '__main__':
    main()
    datetime.time
