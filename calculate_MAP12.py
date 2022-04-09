import numpy as np
import pandas as pd
from numpy import float64
from tqdm.auto import tqdm
from typing import List, Dict


def calculate_apk(actual_items, predicted_items: List, k: int = 10)->float:
    '''
    AP@Kを計算する関数。
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : float
            The average precision at k over the input lists
    '''
    # もし予測アイテムがkより多ければ、レコメンドアイテム数の調節。
    if len(predicted_items) > k:
        predicted_items = predicted_items[:k]

    # 初期値を設定
    score = 0.0
    num_hits = 0.0

    # レコメンドアイテム1つ毎に、当たってるかチェック
    for i, predicted_item in enumerate(predicted_items):
        # もし当たっていれば、カウント
        if (predicted_item in actual_items) and (predicted_item not in predicted_items[:i]):
            num_hits += 1.0
            score += num_hits / (i+1.0)
    # AP@Kの算出
    ## もし実測値がなければ、AP@Kは0.0
    if len(actual_items) == 0:
        return 0.0
    ap_at_k = score / min(len(actual_items), k)

    return ap_at_k

def calculate_mapk(actual, predicted:List[List], k:int=10)->float64:
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    # 各ユーザのAP@kを保存するListをInitialize
    list_ap_k = []

    for a, p in zip(actual, predicted):
        
        ap_k = calculate_apk(a, p, k)
        list_ap_k.append(ap_k)
    # MAP@kを算出
    map_k = np.mean(list_ap_k)

    # 上記の処理をリスト内包表記で書くと、
    # map_k = np.mean([calculate_apk(a, p, k) for a, p in zip(actual, predicted)])

    return map_k

    
def main():
    pass


if __name__ == '__main__':
    pass
