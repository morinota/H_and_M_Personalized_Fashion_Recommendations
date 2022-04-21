
from typing import List, Tuple
import pandas as pd
from scripts.dataset import DataSet
from utils.useful_func import iter_to_str
from collections import defaultdict
import surprise 


class LSTM:
    # クラス変数の定義
    INPUT_DIR = r"input"
    DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

    def __init__(self, transaction_train: pd.DataFrame, dataset:DataSet) -> None:
    # インスタンス変数(属性の初期化)
        self.article_df = dataset.dfi
        self.transaction_train = transaction_train

    def preprocessing(self):
        # article_dfの操作
        df = self.article_df
        df = df.drop(columns = ['product_type_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name',
                        'perceived_colour_master_name', 'index_name', 'index_group_name', 'section_name', 
                        'garment_group_name', 'prod_name', 'department_name', 'detail_desc'])
        temp = df.rename(
    columns={'article_id': 'item_id:token', 'product_code': 'product_code:token', 'product_type_no': 'product_type_no:float',
             'product_group_name': 'product_group_name:token_seq', 'graphical_appearance_no': 'graphical_appearance_no:token', 
             'colour_group_code': 'colour_group_code:token', 'perceived_colour_value_id': 'perceived_colour_value_id:token', 
             'perceived_colour_master_id': 'perceived_colour_master_id:token', 'department_no': 'department_no:token', 
             'index_code': 'index_code:token', 'index_group_no': 'index_group_no:token', 'section_no': 'section_no:token', 
             'garment_group_no': 'garment_group_no:token'})

        df = self.transaction_train
        df['t_dat'] = pd.to_datetime
        