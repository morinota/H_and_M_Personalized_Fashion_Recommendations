import implicit
from typing import Dict, List, Tuple
import pandas as pd
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
import scipy.sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import os

INPUT_DIR = r"input"
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


class MatrixFactrization:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.transaction_train = transaction_train
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.hyper_params = {}

    def _count_all_unique_user_and_item(self):
        """Rating Matrixのindexと、customer_id、article_idの対応表を作る。
        """
        self.ALL_USERS = self.dataset.dfu['customer_id_short'].unique(
        ).tolist()  # ユーザidのユニーク値のリスト
        self.ALL_ITEMS = self.dataset.dfi['article_id'].unique(
        ).tolist()  # アイテムidのユニーク値のリスト

    def _add_originalId_item_and_user(self):
        '''
        # ユーザーとアイテムの両方に0から始まる自動インクリメントのidを割り当てる関数
        '''
        # key:0から始まるindex, value:ユーザidのdict
        self.user_ids = dict(list(enumerate(self.ALL_USERS)))
        # key:0から始まるindex, value:アイテムidのdict
        self.item_ids = dict(list(enumerate(self.ALL_ITEMS)))

        # 辞書内包表記で、keyとvalueをいれかえてる...なぜ?? =>mapメソッドを使う為.
        user_map = {u: uidx for uidx, u in self.user_ids.items()}
        item_map = {i: iidx for iidx, i in self.item_ids.items()}

        # mapメソッドで置換.
        # 引数にdictを指定すると、keyと一致する要素がvalueに置き換えられる.
        # customer_id : ユーザと一意に定まる文字列, user_id：0～ユニーク顧客数のindex
        self.transaction_train['user_id'] = self.transaction_train['customer_id_short'].map(
            user_map)
        # article_id : アイテムと一意に定まる文字列, item_id：0～ユニークアイテム数のindex
        self.transaction_train['item_id'] = self.transaction_train['article_id'].map(
            item_map)

    def _get_rating_matrix(self):
        """トランザクションデータから評価行列を作成する関数
        """
        # COO形式で、ユーザ×アイテムの疎行列を生成.
        row = self.transaction_train['user_id'].values  # 行インデックス
        col = self.transaction_train['item_id'].values  # 列インデックス
        # 値 (トランザクションの総数, １の配列)
        data = np.ones(self.transaction_train.shape[0])
        # => あれ？重複含んでない？？

        # 元の疎行列を生成.COO = [値、(行インデックス、列インデックス)]
        self.rating_matrix_coo = scipy.sparse.coo_matrix((data, (row, col)), shape=(
            len(self.ALL_USERS), len(self.ALL_ITEMS)))
        # coo_matrixは同じ座標を指定すると、要素が加算される性質がある。
        # =>各要素が購買回数の、implictな評価行列の完成！

        # csr形式も保存しておく
        self.rating_matrix_csr = self.rating_matrix_coo.tocsr()

    def preprocessing(self):
        """Matrix Factrizationの為の前処理を実行するメソッド.
        """
        self._count_all_unique_user_and_item()
        self._add_originalId_item_and_user()
        self._get_rating_matrix()

    def fit(self, hyper_params: Dict = {'factors': 5,
                                 'iterations': 3,
                                 'regularization': 0.01,
                                 'confidence': 50}):
        """ALSでMatrix Factrizationを実行するメソッド。

        Parameters
        ----------
        hyper_params : Dict, optional
            _description_, by default {}
        """

        # ALSのハイパーパラメータの設定:
        self.hyper_params = hyper_params

        # モデルのInitialize
        self.model = AlternatingLeastSquares(
            factors=self.hyper_params['factors'],
            iterations=self.hyper_params['iterations'],
            regularization=self.hyper_params['regularization'],  # 正則化項のb
            random_state=42
        )
        # 学習
        # Confidenceのパラメータは直接、Rating Matrixへ掛け合わせて設定する。
        self.model.fit(self.hyper_params['confidence'] * self.rating_matrix_coo,
                       show_progress=True)

    
    def get_feature_vectors(self):

        # 学習後、推定されたUser MatrixとItem Matrix(ndarray型)を保存
        self.user_matrix = self.model.user_factors
        self.item_matrix = self.model.item_factors
        print(type(self.user_matrix))
     
        # 長さを確認
        print(self.item_matrix.shape)
        print(len(self.ALL_USERS))
        print(self.item_matrix.shape)
        print(len(self.ALL_ITEMS))

        # DataFrameに加工する
        self.user_matrix_df = pd.DataFrame(
            data=self.user_matrix, index=self.ALL_USERS, 
            columns=['user潜在変数1', 'user潜在変数2', 'user潜在変数3', 'user潜在変数4', 'user潜在変数5']
        )
        self.item_matrix_df = pd.DataFrame(
            data=self.item_matrix, index=self.ALL_ITEMS, 
            columns=['item潜在変数1', 'item潜在変数2', 'item潜在変数3', 'item潜在変数4', 'item潜在変数5']
        )
        del self.user_matrix, self.item_matrix

        # indexに埋まっているcustomer_id_short, article_idを掘り起こす
        self.user_matrix_df = self.user_matrix_df.reset_index()
        self.item_matrix_df = self.item_matrix_df.reset_index()

        # 掘り起こしたindexのカラム名を変更しておく
        self.user_matrix_df.rename(columns={'index':'customer_id_short'}, inplace=True)
        self.item_matrix_df.rename(columns={'index':'article_id'}, inplace=True)
        
        # 特徴量としてエクスポート
        features_dir = os.path.join(DRIVE_DIR, 'feature')
        self.user_matrix_df.to_csv(os.path.join(features_dir, 'user_matrix_features.csv'), index=False)
        self.item_matrix_df.to_csv(os.path.join(features_dir, 'item_matrix_features.csv'), index=False)

        
        
    def _predict(self):
        preds = []
        batch_size = 2000  # バッチサイズ人のユーザのレコメンドを一度に得る。(一斉は無理?)
        # ユーザ数分のindexを格納した行列を生成.
        all_user_ids = np.arange(len(self.ALL_USERS))
        # 各ユーザ毎に、各アイテムのp\hatを算出。
        for start_id in tqdm(range(0, len(all_user_ids), batch_size)):
            # レコメンド対象のユーザidの配列
            batch_user_ids = all_user_ids[start_id: start_id + batch_size]
            # あるユーザ達における、全アイテムの評価値のベクトル?を算出?
            pred_item_ids, pred_scores = self.model.recommend(
                userid=batch_user_ids,
                # Rating Matrixの実測値
                user_items=self.rating_matrix_csr[batch_user_ids],
                N=12,
                filter_already_liked_items=True
            )

            # バッチ内の各ユーザに対して、user_id=>customer_id_short, item_ids=article_id
            for i, userid in enumerate(batch_user_ids):
                # user_id=>customer_id_shortに変換
                customer_id = self.user_ids[userid]
                # バッチ内の各ユーザのレコメンドアイテムを取得
                user_items = pred_item_ids[i]
                # item_ids=>article_idに変換
                article_ids = [self.item_ids[item_id]
                               for item_id in user_items]

                # レコメンド結果を格納
                preds.append((customer_id, ' '.join(
                    [str(x).zfill(10) for x in article_ids])))

        df_preds = pd.DataFrame(
            preds, columns=['customer_id_short', 'prediction'])

        return df_preds

    def create_reccomendation(self):
        df_preds = self._predict()
        df_sub = pd.merge(df_preds,
                          self.dataset.df_sub[['customer_id_short', 'customer_id']],
                          on='customer_id_short',
                          how='left')

        return df_sub


if __name__ == '__main__':
    pass
