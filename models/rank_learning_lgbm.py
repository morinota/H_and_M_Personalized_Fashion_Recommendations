from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm
import os
from lightgbm.sklearn import LGBMRanker
from pathlib import Path

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


class RankLearningLgbm:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet, val_week_id: int) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.transaction_train = transaction_train
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.hyper_params = {}
        self.val_week_id = val_week_id

    def _create_df_1w_to4w(self):
        self.date_minus = timedelta(days=7*(105 - self.val_week_id))
        # トランザクション
        df = self.dataset.df.copy()
        df['article_id'] = df['article_id'].astype('int')
        self.df = df

        self.df_1w = df[df['t_dat'] >= (
            pd.to_datetime('2020-09-15') - self.date_minus)].copy()
        self.df_2w = df[df['t_dat'] >= (
            pd.to_datetime('2020-09-07') - self.date_minus)].copy()
        self.df_3w = df[df['t_dat'] >= (
            pd.to_datetime('2020-08-31') - self.date_minus)].copy()
        self.df_4w = df[df['t_dat'] >= (
            pd.to_datetime('2020-08-24') - self.date_minus)].copy()

    def _load_feature_data(self):
        self.item_features = pd.read_parquet(os.path.join(
            DRIVE_DIR, 'input/item_features.parquet')).reset_index()
        self.user_features = pd.read_parquet(os.path.join(
            DRIVE_DIR, 'input/user_features.parquet')).reset_index()
        # customer_id_shortカラムを作成
        print('a')
        self.user_features["customer_id_short"] = self.user_features["customer_id"].apply(
            lambda s: int(s[-16:], 16)).astype("uint64")
        print('a')

    def _preprocessing_user_feature(self):
        self.user_features[['club_member_status', 'fashion_news_frequency']] = (
            self.user_features[['club_member_status',
                                'fashion_news_frequency']]
            .apply(lambda x: pd.factorize(x)[0])).astype("uint64")

    def _merge_user_item_feature_to_transactions(self):
        self.df = self.df.merge(self.user_features, on=('customer_id_short'))
        self.df = self.df.merge(self.item_features, on=('article_id'))
        # 降順で並び変え
        self.df.sort_values(['t_dat', 'customer_id'],
                            inplace=True, ascending=False)

    def _create_train_and_valid(self):
        N_ROWS = 1_000_000
        self.train = self.df.loc[self.df.t_dat <= (
            pd.to_datetime('2020-09-15') - self.date_minus)].iloc[:N_ROWS]
        self.valid = self.df.loc[self.df.t_dat >= (
            pd.to_datetime('2020-09-16') - self.date_minus)]

        # delete transactions to save memory
        del self.df

    def preprocessing(self):
        """Matrix Factrizationの為の前処理を実行するメソッド.
        """
        self._create_df_1w_to4w()
        print("g")
        self._load_feature_data()
        print("g")

        self._preprocessing_user_feature()
        print("g")
        self._merge_user_item_feature_to_transactions()
        print("g")
        self._create_train_and_valid()
        print("g")

    def _create_purchased_dict(self):
        """過去の各ユーザのトランザクション情報をまとめる。
        この情報を元に、各ユーザに対するレコメンド「候補」n個を生成する。
        """

        def __create_purchased_dict(df_iw: pd.DataFrame):
            # 一定期間のトランザクションログを受け取り、{ユーザid: {購入したアイテムid:購入回数}}を生成？？
            # 結果格納用のDictをInitialize
            purchase_dict_iw = {}

            # 各トランザクションログ毎に繰り返し処理
            for i, x in enumerate(zip(df_iw['customer_id_short'], df_iw['article_id'])):
                cust_id, art_id = x

                # ユーザが、まだdictに含まれていなければ...
                if cust_id not in purchase_dict_iw:
                    # dictのkeyとして登録＝ユーザがその期間にトランザクションした事を意味する。
                    purchase_dict_iw[cust_id] = {}
                # アイテムが、まだ対象ユーザの購入dictに含まれていなければ...
                if art_id not in purchase_dict_iw[cust_id]:
                    # dictのkeyとして登録
                    purchase_dict_iw[cust_id][art_id] = 0

                # dictの、ユーザkeyのdictの、カウントを+1する。
                purchase_dict_iw[cust_id][art_id] += 1

            # また、対象期間内で、最もよく売れたアイテム、上位12個も用意しておく。
            dummy_list_iw = list(
                (df_iw['article_id'].value_counts()).index)[:100]

            return purchase_dict_iw, dummy_list_iw

        self.purchase_dict_4w, self.dummy_list_4w = __create_purchased_dict(
            self.df_4w)
        self.purchase_dict_3w, self.dummy_list_3w = __create_purchased_dict(
            self.df_3w)
        self.purchase_dict_2w, self.dummy_list_2w = __create_purchased_dict(
            self.df_2w)
        self.purchase_dict_1w, self.dummy_list_1w = __create_purchased_dict(
            self.df_1w)

    def __prepare_candidates(self, customers_id, n_candidates: int = 100):
        """各ユーザ毎に、各ユーザの過去の購買記録に基づいて、全アイテムの中から購入しそうなアイテムn(=ex. 1000)個を抽出し、候補として渡す。
        その「候補」をランク付けする事でレコメンドを達成する。
        過去の購買記録にないユーザに対しては、代替の方法で「候補」n個を用意する。(学習時には不要。推論時には必要)

        df - basically, dataframe with customers(customers should be unique)。ユーザidのseriesを渡す。
        """
        # 結果のDictをInitialize:Dict[ユーザid, 「候補」リスト]
        prediction_dict = {}
        # 直近2週間の売上上位アイテムn_candidates個を、リストで用意しておく。
        dummy_list = list((self.df_2w['article_id'].value_counts()).index)[
            :n_candidates]

        # 各ユーザ毎に繰り返し処理
        for i, cust_id in tqdm(enumerate(customers_id)):
            # もし対象ユーザが、直近1weekで何かアイテム購入していれば、それを元に「候補」を作る。
            if cust_id in self.purchase_dict_1w:
                # 購入アイテムid&購入回数の[[]]を購入回数の降順でソート。
                l = sorted((self.purchase_dict_1w[cust_id]).items(
                ), key=lambda x: x[1], reverse=True)
                # アイテムidのみを残す。
                l = [y[0] for y in l]
                # もし、アイテムidの個数が「候補」数より大きければ...
                if len(l) > n_candidates:
                    # 削る
                    s = l[:n_candidates]
                else:
                    # 直近1週間の上位売上アイテムから補完する。
                    s = l+self.dummy_list_1w[:(n_candidates-len(l))]

            # 1weekで購入履歴がなく、2weekで購入履歴があれば...
            elif cust_id in self.purchase_dict_2w:
                l = sorted((self.purchase_dict_2w[cust_id]).items(
                ), key=lambda x: x[1], reverse=True)
                l = [y[0] for y in l]
                if len(l) > n_candidates:
                    s = l[:n_candidates]
                else:
                    s = l+self.dummy_list_2w[:(n_candidates-len(l))]
            # 1, 2weekで購入履歴がなく、3weekで購入履歴があれば...
            elif cust_id in self.purchase_dict_3w:
                l = sorted((self.purchase_dict_3w[cust_id]).items(
                ), key=lambda x: x[1], reverse=True)
                l = [y[0] for y in l]
                if len(l) > n_candidates:
                    s = l[:n_candidates]
                else:
                    s = l+self.dummy_list_3w[:(n_candidates-len(l))]
            # 1, 2, 3weekで購入履歴がなく、4weekで購入履歴があれば...
            elif cust_id in self.purchase_dict_4w:
                l = sorted((self.purchase_dict_4w[cust_id]).items(
                ), key=lambda x: x[1], reverse=True)
                l = [y[0] for y in l]
                if len(l) > n_candidates:
                    s = l[:n_candidates]
                else:
                    s = l+self.dummy_list_4w[:(n_candidates-len(l))]
            # 1, 2, 3, 4weekで購入履歴がなければ...
            else:
                # 直近2週間の売上上位アイテムn_candidates個を、リストで用意しておく。
                s = dummy_list

            # 対象ユーザの「候補」アイテム群n_candidates個として、dictに格納
            prediction_dict[cust_id] = s

        # 渡された全てのユニークユーザに対して、「候補」を作成できたら...
        # ＝＞user_idと「候補」をそれぞれリストで取得。
        k = list(map(lambda x: x[0], prediction_dict.items()))
        v = list(map(lambda x: x[1], prediction_dict.items()))

        # DataFrameとして保存.(user_id, [候補アイテムのリスト])
        negatives_df = pd.DataFrame({'customer_id_short': k, 'negatives': v})
        # explodeカラムで、[候補アイテムのリスト]をレコードに展開する！他のカラムの要素は複製される。
        negatives_df = negatives_df.explode('negatives')
        # 「候補」アイテムのカラムをRename
        negatives_df.rename(columns={'negatives': 'article_id'}, inplace=True)

        return negatives_df

    def _create_label_column(self):
        self.train: pd.DataFrame
        # take only last 15 transactions

        # まずトランザクションログに、0~len(train)の通し番号を付ける。
        self.train['rank'] = range(len(self.train))
        # assign()メソッドで新規カラムを追加or既存カラムに値を代入
        # ここでは、rmカラムを新規に追加してる。
        self.train = self.train.assign(
            rn=self.train.groupby(['customer_id_short'])['rank']
            .rank(method='first', ascending=False))
        self.train = self.train.query("rn <= 15")
        self.train.drop(columns=['price', 'sales_channel_id'], inplace=True)
        self.train.sort_values(['t_dat', 'customer_id_short'], inplace=True)

        self.train['label'] = 1

        del self.train['rank']
        del self.train['rn']

        self.train.sort_values(['t_dat', 'customer_id_short'], inplace=True)

    def _append_negatives_to_positives_using_lastDate_fromTrain(self):
        # 各ユーザに対して、学習データ期間の最終購入日を取得する。
        last_dates = (
            self.train
            .groupby('customer_id_short')['t_dat']
            .max()
            .to_dict()
        )

        # 各ユーザに対して、「候補」アイテムをn個取得する。(transaction_dfっぽい形式になってる!)
        self.negatives_df = self.__prepare_candidates(
            customers_id=self.train['customer_id_short'].unique(), n_candidates=100)
        # negativeなレコードのt_datは、last_datesで穴埋めする。
        self.negatives_df['t_dat'] = self.negatives_df['customer_id_short'].map(
            last_dates)
        # データ型を一応変換しておく。
        self.negatives_df['article_id'] = self.negatives_df['article_id'].astype(
            'int')
        # negatives_df(<=候補アイテム)にユーザ特徴量＆アイテム特徴量を結合する。
        self.negatives_df = (
            self.negatives_df
            .merge(self.user_features, on=('customer_id_short'))
            .merge(self.item_features, on=('article_id'))
        )
        # negatives_dfのLabelカラムを0にする。(重複ない??)
        self.negatives_df['label'] = 0

    def _merge_train_and_negatives(self):

        # 縦にくっつける...重複ない??
        self.train = pd.concat([self.train, self.negatives_df])

    def _create_query_data(self):
        """
        学習データにクエリID列を持たせるメソッド。

        """
        # まずクエリIDを持たせる為に、customer_idでソートして、その中でt_datでソート
        self.train.sort_values(['customer_id_short', 't_dat'], inplace=True)

        self.train_baskets = self.train.groupby(['customer_id_short'])[
            'article_id'].count().values

        # 検証用データに対しても同様に、クエリデータを生成する。
        self.valid.sort_values(['customer_id_short', 't_dat'], inplace=True)
        self.valid_baskets = self.valid.groupby(['customer_id_short'])[
            'article_id'].count().values

    def fit(self):
        """Fit lightgbm ranker model
        """
        self._create_purchased_dict()
        self._create_label_column()
        self._append_negatives_to_positives_using_lastDate_fromTrain()
        self._merge_train_and_negatives()
        self._create_query_data()

        # データセットを用意

        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3, 5],
            'boosting_type': 'gbdt',
            'max_depth': 15,
            'n_estimators': 1000,
            'verbose': 10,
            'importance_type': 'gain'
        }
        self.ranker = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                boosting_type="dart",
                max_depth=7,
                n_estimators=300,
                importance_type='gain',
                verbose=10,
                ndcg_eval_at = [3,5]
        )
        # 特徴量とターゲットを分割
        X_train = self.train.drop(
            columns=['t_dat', 'customer_id', 'customer_id_short', 'article_id', 'label', 'week'])
        y_train = self.train['label']
        # 特徴量のカラム名を保存
        self.feature_names = X_train.columns
        # X_valid = self.valid.drop(
        #     columns=['t_dat', 'customer_id', 'customer_id_short', 'article_id', 'label', 'week'])
        # y_valid = self.valid['label']
        # 学習
        self.ranker = self.ranker.fit(
            X=X_train,
            y=y_train,
            group=self.train_baskets,
        )

    def _prepare_prediction(self):
        self.sample_sub = self.dataset.cid

        self.candidates = self.__prepare_candidates(
            self.sample_sub['customer_id_short'].unique(), 12)
        self.candidates['article_id'] = self.candidates['article_id'].astype(
            'int')
        self.candidates = (
            self.candidates
            .merge(self.user_features, on=('customer_id_short'))
            .merge(self.item_features, on=('article_id'))
        )

    def _predict_using_batches(self):
        """Predict using batches, otherwise doesn't fit into memory.
        """
        # 予測値のリストをInitialize
        self.preds = []
        # batchサイズ分ずつ、予測していく.
        batch_size = 1_000_000
        for bucket in tqdm(range(0, len(self.candidates), batch_size)):
            # 特徴量を用意。
            X_pred = self.candidates.iloc[bucket: bucket +
                                          batch_size][self.feature_names]
            # モデルに特徴量を入力して、出力値を取得
            outputs = self.ranker.predict(X=X_pred)
            # 予測値のリストに追加
            self.preds.append(outputs)

    def _prepare_submission(self):
        self.preds = np.concatenate(self.preds)

        self.candidates['preds'] = self.preds
        self.preds = self.candidates[[
            'customer_id_short', 'article_id', 'preds']]
        self.preds.sort_values(
            ['customer_id_short', 'preds'], ascending=False, inplace=True)
        self.preds = (
            self.preds.groupby('customer_id_short')[
                ['article_id']].aggregate(lambda x: x.tolist())
        )
        # 提出用にレコメンドアイテムの体裁を整える。
        self.preds['article_id'] = self.preds['article_id'].apply(
            lambda x: ' '.join(['0'+str(k) for k in x]))

        # モデルでレコメンドしきれていないユーザを補完
        self.preds = self.sample_sub[['customer_id_short']].merge(
            self.preds
            .reset_index()
            .rename(columns={'article_id': 'prediction'}), how='left')
        self.preds['prediction'].fillna(
            ' '.join(['0'+str(art) for art in self.dummy_list_2w]), inplace=True)

    def create_reccomendation(self) -> pd.DataFrame:
        self._prepare_prediction()
        self._predict_using_batches()
        self._prepare_submission()

        return self.preds


if __name__ == '__main__':
    pass
