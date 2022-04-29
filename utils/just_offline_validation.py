from utils.calculate_MAP12 import apk, mapk, mean_precision_k
from typing import List, Tuple
import pandas as pd
from config import Config
import numpy as np

def offline_validation(val_df:pd.DataFrame, pred_df:pd.DataFrame)->Tuple[float,float]:

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
    ap12 = mapk(actual, predicted, return_apks=True) # ap12のArray
    # MAP@kを算出
    map12 = round(np.mean(ap12), 6)

    # 各ユーザのprecision@12を算出
    p12 = mean_precision_k(actual, predicted, return_pks=True)
    mean_p12 = round(np.mean(p12), 6)

    return map12, mean_p12