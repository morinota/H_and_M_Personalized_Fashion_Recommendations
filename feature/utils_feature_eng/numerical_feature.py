import pandas as pd
import numpy as np
from pyrsistent import inc
from tqdm import tqdm
import math
from my_class.dataset import DataSet
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Union


def create_is_null_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    """is_nullカラムを追加する関数

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    column_name : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    df[f'{column_name}_is_null'] = df[column_name].isna()
    # boolを0/1に変換
    df[f'{column_name}_is_null'] = df[f'{column_name}_is_null'] * 1
    
    return df

def _create_is_zero_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    pass

def _create_is_not_zero_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    pass

def _create_over_0point5_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    
    df[f'{column_name}_is_null']
    pass

def drop_same_meaning_columns(df:pd.DataFrame)->pd.DataFrame:

    return df

def extract_categorical_columns(df:pd.DataFrame)->pd.DataFrame:

    categorical_columns = df.select_dtypes(include=['object']).columns
    print(categorical_columns)

    return df[categorical_columns]

def create_numerical_to_categorical_bin_column(df:pd.DataFrame, column_name):
    

