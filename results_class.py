import os
from typing import Dict
import pandas as pd
import numpy as np
import scipy.sparse
from glob import glob


class Results:
    # クラス変数の定義(必要あれば)

    def __init__(self) -> None:
        # インスタンス変数(属性の初期化)
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.INPUT_DIR = r"input"
        self.DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'
        pass

    def read_val_data(self):
        """検証用のレコメンド結果を読み込むMethod
        """
        # ファイルパスを用意
        csv_sub = os.path.join(self.INPUT_DIR, 'sample_submission.csv')
        val_dir = os.path.join(self.DRIVE_DIR, 'val_results_csv')

        # 提出用のサンプル
        self.df_sub = pd.read_csv(csv_sub)

        # customer_idカラムのみのpd.DataFrameを作っておく(たぶん色々便利なので)
        self.df_sub["customer_id_short"] = pd.DataFrame(
            self.df_sub["customer_id"].apply(lambda s: int(s[-16:], 16))).astype("uint64")
        self.cid = pd.DataFrame(self.df_sub["customer_id_short"])

        # globで対象のファイルパスをリストで取得する。
        self.all_csv_path_list = glob(os.path.join(val_dir, '*.csv'))
        # ファイル名のリストを取得
        self.file_name_list = [p.split('/')[-1]
                               for p in self.all_csv_path_list]
        # アプローチ名のリストを取得('val_○○.csv' もしくは、'sub_○○.csv'の○○のみを抽出。)
        self.approach_names_list = [
            p.split('.')[0][4:] for p in self.file_name_list]

        print(self.approach_names_list)

        # 各レコメンド結果を読み込み
        self.results_dict: Dict[str, pd.DataFrame] = {}
        for approach_name, file_path in zip(self.approach_names_list, self.all_csv_path_list):
            result_df = pd.read_csv(file_path)
            self.results_dict[approach_name] = result_df

    def join_results_all_approaches(self):
        """検証用のレコメンド結果を1つのDataFrameに結合する関数。
        """

        # 順番にマージしていく
        sub_df = self.df_sub.copy()
        for name, df in self.results_dict.items():
            df = df[['customer_id', 'prediction']].rename(columns={'prediction':f'{name}'})
            sub_df = pd.merge(sub_df, df, how='left',
                              on='customer_id')
        
        self.df_sub = sub_df
        print(self.df_sub.columns)


def main():
    val_results = Results()
    val_results.read_val_data()
    val_results.join_results_all_approaches()


if __name__ == '__main__':
    main()
