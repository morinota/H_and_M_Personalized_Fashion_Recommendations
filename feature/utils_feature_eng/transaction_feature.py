from typing import List

from sqlalchemy import column
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import itertools

def create_popular_ranking_in_transaction_log(self, df_transaction:pd.DataFrame, target_column:str='article_id'):
    # target_column各要素の売り上げをカウント
    df_popular_target = df_transaction.groupby(
        by=[target_column])['customer_id_short'].count().reset_index()

    # カラム名を変更
    df_popular_target.rename(
        columns={'customer_id_short': 'counts'},
        inplace=True
    )

    return df_popular_target