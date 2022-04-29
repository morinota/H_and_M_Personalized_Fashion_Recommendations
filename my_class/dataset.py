import os
from cv2 import INPAINT_NS
import pandas as pd

import numpy as np
import scipy.sparse


class DataSet:
    # クラス変数の定義
    DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'
    INPUT_DIR = os.path.join(DRIVE_DIR, 'input')

    def __init__(self) -> None:
        # インスタンス変数(属性の初期化)
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.df_val: pd.DataFrame
        pass

    def read_data(self, c_id_short: bool = True):

        # ファイルパスを用意
        csv_train = os.path.join(DataSet.INPUT_DIR, 'transactions_train.csv')
        csv_sub = os.path.join(DataSet.INPUT_DIR, 'sample_submission.csv')
        csv_users = os.path.join(DataSet.INPUT_DIR, 'customers.csv')
        csv_items = os.path.join(DataSet.INPUT_DIR, 'articles.csv')

        # データをDataFrame型で読み込み
        if c_id_short == True:
            # 実際の購買記録の情報
            self.df = pd.read_parquet(os.path.join(
                DataSet.DRIVE_DIR, 'transactions_train.parquet'))
            # dfのcustomer_idはshort版に加工されてるから、カラム名を変更しておく
            self.df.rename(
                columns={'customer_id': 'customer_id_short'}, inplace=True)

            # dfのarticle_idを文字列に為ておく?
            # 各顧客の情報(メタデータ)
            self.dfu = pd.read_parquet(os.path.join(
                DataSet.DRIVE_DIR, 'customers.parquet'))
            self.dfu.rename(
                columns={'customer_id': 'customer_id_short'}, inplace=True)
            # 各商品の情報(メタデータ)
            self.dfi = pd.read_parquet(os.path.join(
                DataSet.DRIVE_DIR, 'articles.parquet'))
        else:
            self.df = pd.read_csv(csv_train, dtype={'article_id': str},
                                  parse_dates=['t_dat']  # datetime型で読み込み
                                  )
            self.dfu = pd.read_csv(csv_users)  # 各顧客の情報(メタデータ)
            self.dfi = pd.read_csv(
                csv_items, dtype={'article_id': str})  # 各商品の情報(メタデータ)

            # customer_id_shortカラムを生成
            self.df['customer_id_short'] = self.df["customer_id"].apply(lambda s: int(s[-16:], 16)).astype("uint64")
            self.dfu['customer_id_short'] =self.dfu["customer_id"].apply(lambda s: int(s[-16:], 16)).astype("uint64")

        # price カラムを×10^3しておく...その方が、小数点以下と整数で分けやすい??
        self.df['price'] = self.df['price'] * (10 **3)

        # 提出用のサンプル
        self.df_sub = pd.read_csv(csv_sub)
        

        # customer_idカラムのみのpd.DataFrameを作っておく(たぶん色々便利なので)
        self.df_sub["customer_id_short"] = pd.DataFrame(
            self.df_sub["customer_id"].apply(lambda s: int(s[-16:], 16))).astype("uint64")
        self.cid = pd.DataFrame(self.df_sub["customer_id_short"])

    def read_data_sampled(self, sampling_percentage: float = 5):
        # ファイルパスを用意
        sampled_data_dir = os.path.join(DataSet.INPUT_DIR, 'sampling_dir')
        path_transactions = os.path.join(
            sampled_data_dir, f'transactions_train_sample{sampling_percentage}.csv.gz')
        path_article = os.path.join(
            sampled_data_dir, f'articles_train_sample{sampling_percentage}.csv.gz')
        path_customers = os.path.join(
            sampled_data_dir, f'customers_sample{sampling_percentage}.csv.gz')

        # インスタンス変数として読み込み
        self.df = pd.read_csv(path_transactions,
                              dtype={'article_id': str},
                              parse_dates=['t_dat']  # datetime型で読み込み
                              )
        # price カラムを×10^3しておく...その方が、小数点以下と整数で分けやすい??
        self.df['price'] = self.df['price'] * (10 **3)
        self.dfi = pd.read_csv(path_article, dtype={'article_id': str})
        self.dfu = pd.read_csv(path_customers)
        # df_subはそのまま
        csv_sub = os.path.join(DataSet.INPUT_DIR, 'sample_submission.csv')
        self.df_sub = pd.read_csv(csv_sub)
        # customer_id_shortカラムを作る.
        self.df_sub["customer_id_short"] = pd.DataFrame(
            self.df_sub["customer_id"].apply(lambda s: int(s[-16:], 16))).astype("uint64")

        # customer_idカラムのみのpd.DataFrameを作っておく(たぶん色々便利なので)
        self.cid = pd.DataFrame(self.dfu["customer_id_short"].copy())
        print(self.cid)


def main():

    print(scipy.sparse.coo_matrix)


if __name__ == '__main__':
    main()
