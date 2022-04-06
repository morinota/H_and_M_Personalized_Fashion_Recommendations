from calculate_MAP12 import calculate_mapk
from collections import defaultdict
import seaborn as sns
from typing import List
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def _iter_to_str(iterable: List[int]):
    '''
    article_idの先頭に0を追加し、各article_idを半角スペースで区切られた文字列を返す関数(提出の様式にあわせる為に必要？)
    '''
    # Listの各要素の先頭に"0"を追加する
    iterable = map(lambda x: str(0) + str(x), iterable)
    # リストの要素を半角スペースで繋いで、文字列に。
    iterable_str = " ".join(iterable)
    return iterable_str


def _blend(bt, w=[], k=12):
    '''
    何かをブレンドする関数
    '''
    if len(w) == 0:
        w = [1] * len(dt)
    preds = []
    for i in range(len(w)):
        preds.append(dt[i].split())
    res = {}
    for i in range(len(preds)):
        if w[i] < 0:
            continue
        for n, v in enumerate(preds[i]):
            if v in res:
                res[v] += (w[i] / (n + 1))
            else:
                res[v] = (w[i] / (n + 1))    
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return ' '.join(res[:k])

def oneweek_holdout_validation():
    pass

def main():
    sample1_list = list(range(5))
    print(_iter_to_str(iterable=sample1_list))


if __name__ == '__main__':
    main()
