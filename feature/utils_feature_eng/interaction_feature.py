from typing import List

from sqlalchemy import column
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import itertools

"""
PolynomialFeaturesクラスは特徴量（または単に変数）のべき乗を求めるもの。
特徴量が複数ある場合には、異なる特徴量間の積も計算する。
"""


def create_interaction_columns(df: pd.DataFrame, target_columns: List[str], interaction_only=True):

    # PolynomialFeaturesインスタンスを生成
    polynominal_features = PolynomialFeatures(
        degree=2,  # 何次の項まで計算するか指定する
        interaction_only=interaction_only,  # Trueにすると、ある特徴量を2乗以上した項が出力されなくなる
        include_bias=False,  # Trueにすると、定数項(=1=x^0=0次項)を出力する
        order='C'
    )
    df_targets = df[target_columns]
    array_interactions = polynominal_features.fit_transform(
        # Arrayで渡す(ReturnもArray)
        X=df_targets.values
    )

    # カラム名を再作成する.(Bias->係数1個->係数2個->...の順)
    new_column_names = target_columns.copy()
    if interaction_only == True:
        all = itertools.combinations(iterable=range(len(target_columns)), r=2)

        for c_name_indices in all:
            column_a = target_columns[c_name_indices[0]]
            column_b = target_columns[c_name_indices[1]]
            new_column_name = f'{column_a}_and_{column_b}'

            new_column_names.append(new_column_name)

    elif interaction_only == False:
        all = itertools.combinations_with_replacement(
            iterable=range(len(target_columns)), r=2)

        for c_name_indices in all:
            column_a = target_columns[c_name_indices[0]]
            column_b = target_columns[c_name_indices[1]]
            if column_a == column_b:
                new_column_name = f'{column_a}_^2'
            else:
                new_column_name = f'{column_a}_and_{column_b}'

            new_column_names.append(new_column_name)

    print(new_column_names)

    # 作成したarray_interactionsとnew_columns_namesを元にDataFrameを生成
    df_interactions = pd.DataFrame(data=array_interactions,
                                   columns=new_column_names
                                   )

    # 元のDataFrameとマージ
    df = pd.merge(
        left=df.drop(columns=target_columns, axis='column'),
        right=df_interactions,
        right_index=True, left_index=True
    )

    return df


if __name__ == '__main__':

    sample_df = pd.DataFrame(
        data=[[2, 3, 5],
              [3, 5, 6],
              [1, 9, 5]],
        columns=['a', 'b', 'c']
    )
    print(sample_df)

    df_new = create_interaction_columns(df=sample_df, target_columns=['b', 'c'],
                                        interaction_only=False)
                                        

    print(df_new)

