from typing import List

from sqlalchemy import column
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import itertools

def create_popular_ranking_in_transaction_log(df_transaction:pd.DataFrame, target_column:str='article_id'):
    # target_column各要素の売り上げをカウント
    df_popular_target = df_transaction.groupby(
        by=[target_column])['customer_id_short'].count().reset_index()

    # カラム名を変更
    df_popular_target.rename(
        columns={'customer_id_short': f'counts_{target_column}'},
        inplace=True
    )
    df_popular_target.sort_values(by=f'counts_{target_column}', ascending=False, inplace=True)
    df_popular_target[f'pop_rank_{target_column}'] = (
        df_popular_target[f'counts_{target_column}'].rank(ascending=False)
    )

    return df_popular_target

def create_popular_ranking_in_transaction_log_with_each_grouping(self, 
df_transaction:pd.DataFrame,
 target_column:str='article_id',
 grouping_column:str='age_bins'):
    # target_column各要素の売り上げをカウント
    df_popular_target = df_transaction.groupby(
        by=[target_column])['customer_id_short'].count().reset_index()

    # カラム名を変更
    df_popular_target.rename(
        columns={'customer_id_short': f'counts_{target_column}'},
        inplace=True
    )
    df_popular_target.sort_values(by='counts', ascending=False, inplace=True)
    df_popular_target[f'pop_rank_{target_column}'] = (
        df_popular_target[f'counts_{target_column}'].rank(ascending=False)
    )

    return df_popular_target