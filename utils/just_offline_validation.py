from multiprocessing.spawn import import_main_path
from calculate_MAP12 import apk, mapk
from typing import List
import pandas as pd
from config import Config
import numpy as np

def offline_validation(val_df:pd.DataFrame, pred_df:pd.DataFrame)->float:

    # レコードの順番をそろえたい...。
    val_df = val_df.sort_values(by='customer_id_short')
    pred_df = pred_df.sort_values(by='customer_id_short')

    # Listで抽出
    actual: List[List[str]] = val_df['article_id'].apply(
        lambda s: [] if pd.isna(s) else s.split())
    predicted: List[List[str]] = pred_df['prediction'].apply(
        lambda s: [] if pd.isna(s) else s.split())

    k = Config.num_recommend_item

    print(f'length of user in actual is...{len(actual)}')
    print(f'length of user in predicted is...{len(predicted)}')

    # 各ユーザのAP@kを算出(後半のヒストグラム作成の為に)
    ap12 = mapk(actual, predicted, return_apks=True)
    # MAP@kを算出
    map12 = round(np.mean(ap12), 6)

    return map12