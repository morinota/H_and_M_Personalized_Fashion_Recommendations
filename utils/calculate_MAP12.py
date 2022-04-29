import numpy as np
import pandas as pd
from numpy import float64
from tqdm.auto import tqdm
from typing import List, Dict


def p_k(actual: List[str], predicted: List[str], k=12) -> float:
    """ある一人のユーザに対するレコメンドのPrecision@Kを出力する関数

    Parameters
    ----------
    actual : List[str]
        _description_
    predicted : List[str]
        _description_
    k : int, optional
        _description_, by default 12

    Returns
    -------
    float
        ある一人のユーザに対するレコメンドのPrecision@K
    """
    # もしレコメンドアイテム数が多かったら削る.
    if len(predicted) > k:
        predicted = predicted[:k]

    nhits = 0.0
    # 各レコメンドアイテムについて的中しているかチェック。
    for i, p in enumerate(predicted):
        # i番目のpが的中してるなら...
        if p in actual and p not in predicted[:i]:
            # Precision_kを算出。
            nhits += 1.0

    # precison@Kを算出
    precision_at_k = nhits/k

    return precision_at_k


def apk(actual, predicted, k=12):
    """ある一人のユーザに対するレコメンドのAP@Kを出力する関数。

    Parameters
    ----------
    actual : _type_
        _description_
    predicted : _type_
        _description_
    k : int, optional
        _description_, by default 12

    Returns
    -------
    _type_
        _description_
    """
    # もしレコメンドアイテム数が多かったら削る.
    if len(predicted) > k:
        predicted = predicted[:k]
    # スコアの初期値(0~kで足し合わせていく)
    score, nhits = 0.0, 0.0

    # 各レコメンドアイテムについて
    for i, p in enumerate(predicted):
        # i番目のpが的中してる場合のみ、k=iとしてPrecision_kを算出。
        if p in actual and p not in predicted[:i]:
            # Precision_kを算出。
            nhits += 1.0
            precision_at_k = nhits / (i + 1.0)
            # precision_kを足し合わせる。
            score += precision_at_k

    # そもそも実測値がなかったら=ユーザのトランザクションがゼロなら。
    if not actual:
        return 0.0

    # 最終的には、min(k, N(actual))で割った値を返す(MAP@Kの為)
    return score / min(len(actual), k)


def mapk(actual: List[List[str]], predicted: List[List[str]], k: int = 12, return_apks: bool = False):
    assert len(actual) == len(predicted)
    # 各ユーザ毎(actualが1以上の全ユーザ)にAP@Kを算出し、リストに格納。
    apks = [apk(ac, pr, k) for ac, pr in zip(actual, predicted) if 0 < len(ac)]
    if return_apks:
        return apks
    return np.mean(apks)


def mean_precision_k(actual: List[List[str]], predicted: List[List[str]], k: int = 12, return_pks: bool = False):
    assert len(actual) == len(predicted)
    # 各ユーザ毎(actualが少なくとも1以上の全ユーザ)にP@Kを算出し、リストに格納。
    pks = [p_k(ac, pr, k) for ac, pr in zip(actual, predicted) if 0 < len(ac)]
    if return_pks:
        return pks
    return np.mean(pks)


def main():
    pass


if __name__ == '__main__':
    pass
