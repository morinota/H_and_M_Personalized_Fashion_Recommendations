import os
import pandas as pd
import implicit
import numpy as np
import scipy.sparse


class DataSet:
    # クラス変数の定義
    INPUT_DIR = r"input"
    DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

    def __init__(self) -> None:
        # インスタンス変数(属性の初期化)
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        pass

    def read_data(self):

        # ファイルパスを用意
        csv_train = os.path.join(DataSet.INPUT_DIR, 'transactions_train.csv')
        csv_sub = os.path.join(DataSet.INPUT_DIR, 'sample_submission.csv')
        csv_users = os.path.join(DataSet.INPUT_DIR, 'customers.csv')
        csv_items = os.path.join(DataSet.INPUT_DIR, 'articles.csv')

        # データをDataFrame型で読み込み
        # self.df = pd.read_csv(csv_train, dtype={'article_id': str},
        #                       parse_dates=['t_dat'] # datetime型で読み込み
        #                       )  # 実際の購買記録の情報
        self.df = pd.read_parquet(os.path.join(DataSet.DRIVE_DIR, 'transactions_train.parquet'))
        self.df_sub = pd.read_csv(csv_sub)  # 提出用のサンプル
        # self.dfu = pd.read_csv(csv_users)  # 各顧客の情報(メタデータ)
        self.dfu = pd.read_parquet(os.path.join(DataSet.DRIVE_DIR, 'customers.parquet'))  # 各顧客の情報(メタデータ)

        # self.dfi = pd.read_csv(csv_items, dtype={'article_id': str})  # 各商品の情報(メタデータ)
        self.dfi = pd.read_parquet(os.path.join(DataSet.DRIVE_DIR, 'articles.parquet'))  # 各商品の情報(メタデータ)

    def _extract_byDay(self):
        mask = self.df['t_dat'] > '2020-08-21'
        self.df = self.df[mask]

    def _count_all_unique_user_and_item(self):
        self.ALL_ITEMS = self.dfu['customer_id'].unique(
        ).tolist()  # ユーザidのユニーク値のリスト
        self.ALL_USERS = self.dfi['article_id'].unique(
        ).tolist()  # アイテムidのユニーク値のリスト

    def _add_originalId_item_and_user(self):
        '''
        # ユーザーとアイテムの両方に0から始まる自動インクリメントのidを割り当てる関数
        '''
        # key:0から始まるindex, value:ユーザidのdict
        user_ids = dict(list(enumerate(self.ALL_USERS)))
        # key:0から始まるindex, value:アイテムidのdict
        item_ids = dict(list(enumerate(self.ALL_ITEMS)))

        # 辞書内包表記で、keyとvalueをいれかえてる...なぜ?? =>mapメソッドを使う為.
        user_map = {u: uidx for uidx, u in user_ids.items()}
        item_map = {i: iidx for iidx, i in item_ids.items()}

        # mapメソッドで置換.
        # 引数にdictを指定すると、keyと一致する要素がvalueに置き換えられる.
        # customer_id : ユーザと一意に定まる文字列, user_id：0～ユニーク顧客数のindex
        self.df['user_id'] = self.df['customer_id'].map(user_map)
        # article_id : アイテムと一意に定まる文字列, item_id：0～ユニークアイテム数のindex
        self.df['item_id'] = self.df['article_id'].map(item_map)

    def _get_rating_matrix(self):
        '''
        トランザクションデータから評価行列を作成する関数
        '''
        # COO形式で、ユーザ×アイテムの疎行列を生成.
        row = self.df['user_id'].values  # 行インデックス
        col = self.df['item_id'].values  # 列インデックス
        data = np.ones(self.df.shape[0])  # 値 (トランザクションの総数, １の配列)
        # => あれ？重複含んでない？？

        # 元の疎行列を生成.COO = [値、(行インデックス、列インデックス)]
        self.coo_train = scipy.sparse.coo_matrix((data, (row, col)), shape=(
            len(self.ALL_USERS), len(self.ALL_ITEMS)))
        # coo_matrixは同じ座標を指定すると、要素が加算される性質がある。
        # =>各要素が購買回数の、implictな評価行列の完成！

    def _drop_previous_month(self):
        '''
        最後の数日(実験のためのアップ)を除いて、すべてを落とす。
        前の月の情報はあまり意味がない。
        4週間を訓練用データ1~4、最後の1週間を検証用データとして残す。
        '''
        df = self.df
        # object型=>datatime型に変換
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        # 直近4週間のデータをそれぞれトレーニングデータとして抽出
        import datetime
        # ２～１weeks ago
        mask = (df["t_dat"] >= datetime.datetime(2020, 9, 8)) & (
            df['t_dat'] < datetime.datetime(2020, 9, 16))
        self.train1 = df.loc[mask]
        # 3～2 weeks ago
        mask = (df["t_dat"] >= datetime.datetime(2020, 9, 1)) & (
            df['t_dat'] < datetime.datetime(2020, 9, 8))
        self.train2 = df.loc[mask]
        # 4～3 weeks ago
        mask = (df["t_dat"] >= datetime.datetime(2020, 8, 23)) & (
            df['t_dat'] < datetime.datetime(2020, 9, 1))
        self.train3 = df.loc[mask]
        # 5～4 weeks ago
        mask = (df["t_dat"] >= datetime.datetime(2020, 8, 16)) & (
            df['t_dat'] < datetime.datetime(2020, 8, 23))
        self.train4 = df.loc[mask]

        # 1～0 weeks agoを検証用に
        mask = (df["t_dat"] >= datetime.datetime(2020, 9, 16))
        self.val = df.loc[mask]

        del mask

    def preprocessing(self):
        self._extract_byDay()
        self._count_all_unique_user_and_item()
        self._add_originalId_item_and_user()
        self._get_rating_matrix()


def main():
    print(implicit.__version__)
    print(scipy.sparse.coo_matrix)


if __name__ == '__main__':
    main()
