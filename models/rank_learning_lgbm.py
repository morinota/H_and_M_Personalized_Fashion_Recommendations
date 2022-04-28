from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
from sympy import Li
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm
import os
from lightgbm.sklearn import LGBMRanker
from pathlib import Path
from config import Config

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


class RankLearningLgbm:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet, val_week_id: int) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.df = transaction_train
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.hyper_params = {}
        self.val_week_id = val_week_id

    def _create_df_1w_to4w(self):
        """予測期間に対して、過去i week (i=1,...,4)のトランザクションログを抽出して、DataFrameとして保存。
        """
        # val_week_idによって日付をずらす為の変数を用意しておく.
        self.date_minus = timedelta(days=7*(105 - self.val_week_id))
        # 一応、トランザクションデータのカラムのデータ型を調整為ておく
        self.df['article_id'] = self.df['article_id'].astype('int')

        # 直近一週間のトランザクションログ
        self.df_1w = self.df[self.df['t_dat'] >= (
            pd.to_datetime('2020-09-15') - self.date_minus)].copy()
        # 直近2週間のトランザクションログ
        self.df_2w = self.df[self.df['t_dat'] >= (
            pd.to_datetime('2020-09-07') - self.date_minus)].copy()
        # 直近3週間のトランザクションログ
        self.df_3w = self.df[self.df['t_dat'] >= (
            pd.to_datetime('2020-08-31') - self.date_minus)].copy()
        # 直近4週間のトランザクションログ
        self.df_4w = self.df[self.df['t_dat'] >= (
            pd.to_datetime('2020-08-24') - self.date_minus)].copy()

        # 一応、リークがないか確認
        print('df_1w is from {} to {}'.format(
            self.df_1w['t_dat'].max(), self.df_1w['t_dat'].min()))
        print('df_2w is from {} to {}'.format(
            self.df_2w['t_dat'].max(), self.df_2w['t_dat'].min()))
        print('df_3w is from {} to {}'.format(
            self.df_3w['t_dat'].max(), self.df_3w['t_dat'].min()))
        print('df_4w is from {} to {}'.format(
            self.df_4w['t_dat'].max(), self.df_4w['t_dat'].min()))

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
        # 降順(新しい順)で並び変え
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
        self._load_feature_data()
        self._preprocessing_user_feature()
        self._merge_user_item_feature_to_transactions()
        self._create_train_and_valid()

    def _create_purchased_dict(self):
        """過去の各ユーザのトランザクション情報をまとめる。
        この情報を元に、各ユーザに対するレコメンド「候補」n個を生成する。
        """

        def __create_purchased_dict(df_iw: pd.DataFrame)->Tuple[Dict, List]:
            """一定期間のトランザクションログを受け取り、{ユーザid: {購入したアイテムid:購入回数}}を生成？？
            また、対象期間内で、最もよく売れたアイテム、上位12個も生成する。

            Parameters
            ----------
            df_iw : pd.DataFrame
                _description_

            Returns
            -------
            Tuple[Dict, List]
                ({ユーザid: {購入したアイテムid:購入回数}}, 最もよく売れたアイテムのリスト)
            """
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

        del self.df_1w, self.df_3w, self.df_4w

    def __prepare_candidates_original(self, customers_id, n_candidates: int = 100):
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

    def _load_candidate_from_other_recommendation(self):
        approach_name = Config.predict_candidate_way_name
        candidate_path:str = ''
        # 候補データのファイルパスを取得する。
        if Config.run_for_submittion:
            candidate_path = os.path.join(DRIVE_DIR, f'submission_csv/submission_{approach_name}.csv')
        elif Config.run_for_submittion == False:
            candidate_path = os.path.join(
                DRIVE_DIR, 
                'val_results_{}_csv/val_{}.csv'.format(self.val_week_id, approach_name)
                )

        # 読み込み(レコード：ユニークユーザ数、predictionカラムにレコメンド結果が入ってる。)
        candidates_df = pd.read_csv(candidate_path)
        # 'prediction'カラムを変換(str=>List[str]に)
        candidates_df['prediction'] = candidates_df['prediction'].apply(lambda x: x.split(' '))
        # explodeカラムで、[候補アイテムのリスト]をレコードに展開する！他のカラムの要素は複製される。
        candidates_df = candidates_df.explode('prediction')
        # 「候補」アイテムのカラムをRename
        candidates_df.rename(columns={'prediction': 'article_id'}, inplace=True)

        # customer_idカラムを落としておく
        print(candidates_df.columns)
        candidates_df.drop(columns='customer_id', inplace=True)

        # 後はコレをオリジナルとくっつければいいだけだけど...。
        return candidates_df

    def _create_label_column(self):
        """ユーザ毎に# take only last 15 transactions。
        その後、各レコードにlabel=1(すなわち、購入あり)を付与する
        """
        self.train: pd.DataFrame

        # まずトランザクションログ(=新しい順)に、0~len(train)の通し番号を付ける。
        self.train['rank'] = range(len(self.train))
        # assign()メソッドで新規カラムを追加or既存カラムに値を代入
        # ここでは、rmカラムを新規に追加してる。
        self.train = self.train.assign(
            # 各ユーザ毎のrank Seriesに対して、rankメソッドで順位づけする。
            # 降順でソート.
            # method引数で、同一値の処理を指定(通し番号だから問題なくない?)
            # method='first'とすると同一値（重複値）は登場順に順位付け
            rn=self.train.groupby(['customer_id_short'])['rank']
            .rank(method='first', ascending=False))
        # rn が15以下( =ユーザ毎に直近15件)のレコードだけ残す。
        self.train = self.train.query("rn <= 15")
        self.train.drop(columns=['price', 'sales_channel_id'], inplace=True)
        self.train.sort_values(['t_dat', 'customer_id_short'], inplace=True)

        self.train['label'] = 1

        del self.train['rank']
        del self.train['rn']

        self.train.sort_values(['t_dat', 'customer_id_short'], inplace=True)

        # 検証用データに対しても同様(ユーザ毎に直近15件っていう制限はなくていいや)
        self.valid['label'] = 1

    def _append_negatives_to_positives_using_lastDate_fromTrain(self):
        # 各ユーザに対して、学習データ期間の最終購入日を取得する。
        last_dates = (
            self.train
            .groupby('customer_id_short')['t_dat']
            .max()
            .to_dict()
        )

        # 各ユーザに対して、「候補」アイテムをn個取得する。(transaction_dfっぽい形式になってる!)
        if Config.predict_candidate_way_name==None:
            self.negatives_df = self.__prepare_candidates_original(
                customers_id=self.train['customer_id_short'].unique(),
                n_candidates=Config.num_candidate_train)
        else:
            self.negatives_df = self._load_candidate_from_other_recommendation()

        # negativeなレコードのt_datは、last_dates(使うのここだけ?)で穴埋めする。
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

        # 検証用データも同様の手順で、negativeを作る?

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
        self.ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type=Config.boosting_type,
            num_leaves=Config.num_leaves,
            max_depth=Config.max_depth,
            n_estimators=Config.n_estimators,
            bagging_freq=Config.bagging_freq,
            bagging_fraction=Config.bagging_fraction,
            feature_fraction=Config.feature_fraction,
            importance_type='gain',
            verbose=10,
            ndcg_eval_at=[3, 5]
        )
        # 一応、学習用データの最終日を取得し、出力
        last_date_train_data = self.train['t_dat'].max()
        print(f'last date of training data is...{last_date_train_data}aaa')
        print(f'len of train data is {len(self.train)}')

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
        """予測の準備をするメソッド。
        具体的には、
        - sample_submissionを読み込んでおく.
        - 予測する各ユニークユーザに対して、Candidateを用意。
        - 作成されたDataFrame(レコード数= len(unique ser) * n_candidate)に対して、
        アイテム特徴量とユーザ特徴量をマージ
        """
        self.sample_sub = self.dataset.df_sub[['customer_id_short', 'customer_id']].copy()

        # レコメンド候補を用意
        if Config.predict_candidate_way_name == None:
            # NoneだったらオリジナルのCandidate
            self.candidates = self.__prepare_candidates_original(
            self.sample_sub['customer_id_short'].unique(), Config.num_candidate_predict)
        else:
            self.candidates = self._load_candidate_from_other_recommendation()

        # article_idのデータ型をindに
        self.candidates['article_id'] = self.candidates['article_id'].astype(
            'int')

        # ユーザ特徴量＆アイテム特徴量を結合
        self.candidates = (
            self.candidates
            .merge(self.user_features, on=('customer_id_short'))
            .merge(self.item_features, on=('article_id'))
        )

    def _predict_using_batches(self):
        """実際に予測を実行するメソッド。メモリの関係からbatch_size数ずつ入力していく.
        Predict using batches, otherwise doesn't fit into memory.
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
            # -> (ユニークユーザ × num_candidate_predict)個のうち、バッチサイズ分のトランザクションの発生確率を返す?
            # 予測値のリストにndarrayを追加
            self.preds.append(outputs)

    def _prepare_submission(self):
        # 各バッチ毎の予測結果(ndarrayのリスト)を縦に結合
        # ->(ユニークユーザ × num_candidate_predict)個のレコードの発生確率y
        self.preds = np.concatenate(self.preds)
        # 候補アイテムのdf(長さ=(ユニークユーザ × num_candidate_predict))に、予測結果を格納.
        self.candidates['preds'] = self.preds

        # 提出用のカラムのみ抽出.
        self.preds = self.candidates[[
            'customer_id_short', 'article_id', 'preds']]
        # 各ユニークユーザ毎で、発生確率の高いアイテム順にソート
        self.preds.sort_values(
            ['customer_id_short', 'preds'], ascending=False, inplace=True)
        # 「行=ユニークユーザ」「列=レコメンドアイテムのリスト」のDfに変換
        self.preds = (
            self.preds.groupby('customer_id_short')[
                ['article_id']].aggregate(lambda x: x.tolist())
        )
        # 上位12個のみ残す。
        self.preds['article_id'] = self.preds['article_id'].apply(
            lambda x: x[:Config.num_recommend_item])
        # 提出用にレコメンドアイテムの体裁を整える。
        self.preds['article_id'] = self.preds['article_id'].apply(
            lambda x: ' '.join(['0'+str(k) for k in x]))

        # モデルでレコメンドしきれていないユーザを補完
        self.preds = self.sample_sub[['customer_id_short', 'customer_id']].merge(
            self.preds
            .reset_index()
            .rename(columns={'article_id': 'prediction'}), how='left')
        self.preds['prediction'].fillna(
            ' '.join(['0'+str(art) for art in self.dummy_list_2w]), inplace=True)

    def create_reccomendation(self) -> pd.DataFrame:
        self._prepare_prediction()
        self._predict_using_batches()
        self._prepare_submission()

        return self.preds[['customer_id', 'customer_id_short', 'prediction']]


if __name__ == '__main__':
    pass
