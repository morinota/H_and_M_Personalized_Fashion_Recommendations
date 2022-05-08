from copyreg import pickle
from datetime import datetime, timedelta
from re import sub
from turtle import back
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
from models.negative_sampler_class.static_popularity import NegativeSamplerStaticPopularity
import pickle

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'
ITEM_CATEGORICAL_COLUMNS = ['article_id',
                            # 'prod_name',
                            'product_type_name', 'product_group_name',
                            'graphical_appearance_name', 'colour_group_name',
                            'perceived_colour_value_name', 'perceived_colour_master_name',
                            'department_name', 'index_name', 'index_group_name',
                            'section_name', 'garment_group_name'
                            ]


class RankLearningLgbm:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet, val_week_id: int) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.df = transaction_train
        self.val_week_id = val_week_id
        print(len(self.df))
        print('unique user of self.df is {}'.format(
            len(self.df['customer_id_short'].unique())
        ))

    def _extract_non_coldstart_user(self):

        # ユーザアクティビティに関するdfを読み込む
        file_path = os.path.join(DRIVE_DIR, 'input/metadata_customer_id.csv')
        self.user_activity_df = pd.read_csv(file_path)
        self.user_activity_df['customer_id_short'] = self.user_activity_df["customer_id"].apply(
            lambda s: int(s[-16:], 16)).astype("uint64")

        # 'cold_start_status'をトランザクションログにマージする
        self.df = self.df.merge(
            self.user_activity_df[['customer_id_short', 'cold_start_status']],
            on='customer_id_short', how='left'
        )
        # 'cold_start_status' == 'non_cold_start'のユーザのトランザクションログのみ残す
        self.df = self.df.loc[self.df['cold_start_status'] == 'non_cold_start']
        # 'cold_start_status'カラムを落とす
        self.df.drop(columns='cold_start_status', inplace=True)

        print('length of unique user of non coldstart user is {} in train'
              .format(len(self.df['customer_id_short'].unique())))

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
        """事前に作成した特徴量を読み込むメソッド
        """
        # アイテム特徴量
        if Config.use_which_item_features == 'original':
            self.item_features = pd.read_parquet(os.path.join(
                DRIVE_DIR, 'input/item_features.parquet')).reset_index()
        else:
            self.item_features = pd.read_csv(os.path.join(
                DRIVE_DIR, f'input/item_features_{Config.use_which_item_features}.csv')).reset_index()
            # 必要な特徴量のカラムのみ残す.
            self.item_features = self.item_features[[
                'article_id'] + Config.item_basic_feature_names + Config.item_numerical_feature_names+Config.item_one_hot_encoding_feature_names]
        # ユーザ特徴量
        if Config.use_which_item_features == 'original':
            self.user_features = pd.read_parquet(os.path.join(
                DRIVE_DIR, 'input/user_features.parquet')).reset_index()
            # customer_id_shortカラムに変換する
            self.user_features['customer_id_short'] = (
                self.user_features["customer_id"].apply(
                    lambda s: int(s[-16:], 16)).astype("uint64")
            )
            # customer_idカラムを落とす
            self.user_features.drop(columns=['customer_id'], inplace=True)

            # 前処理
            self.user_features[['club_member_status', 'fashion_news_frequency']] = (
                self.user_features[['club_member_status',
                                    'fashion_news_frequency']]
                .apply(lambda x: pd.factorize(x)[0])).astype("uint64")

        else:
            self.user_features = pd.read_csv(os.path.join(
                DRIVE_DIR, f'input/user_features_{Config.use_which_user_features}.csv')).reset_index()
            # 必要なカラムのみ残す
            self.user_features = self.user_features[[
                'customer_id_short'] + Config.user_numerical_feature_names]

        # ラグ特徴量(アイテム)
        self.item_lag_features = {}
        for target_column in Config.item_lag_feature_names_subcategory:
            file_path = os.path.join(
                DRIVE_DIR, f'feature/time_series_item_feature_{target_column}.csv')

            lag_feature = pd.read_csv(file_path)
            # Dictに格納
            self.item_lag_features[target_column] = lag_feature

        # ラグ特徴量(ユーザ)
        self.user_lag_features = {}
        for target_column in Config.user_lag_feature_subcategory:
            file_path = os.path.join(
                DRIVE_DIR, f'feature/time_series_user_feature_{target_column}.csv')

            lag_feature = pd.read_csv(file_path)
            # Dictに格納
            self.user_lag_features[target_column] = lag_feature

        # 潜在変数特徴量
        self.item_hidden_variable_features = pd.read_csv(os.path.join(
            DRIVE_DIR, f'feature/item_matrix_features.csv')).reset_index()
        self.item_hidden_variable_features['article_id'] = self.item_hidden_variable_features['article_id'].astype(
            int)
        self.user_hidden_variable_features = pd.read_csv(os.path.join(
            DRIVE_DIR, f'feature/user_matrix_features.csv')).reset_index()

        # 潜在変数特徴量をマージしておく
        self.item_features = self.item_features.merge(
            self.item_hidden_variable_features, on='article_id', how='left'
        )
        self.user_features = self.user_features.merge(
            self.user_hidden_variable_features, on='customer_id_short', how='left'
        )
        print(f'length of user_features is {len(self.user_features)}')

    def _preprocessing_user_feature(self):
        # self.user_features[['club_member_status', 'fashion_news_frequency']] = (
        #     self.user_features[['club_member_status',
        #                         'fashion_news_frequency']]
        #     .apply(lambda x: pd.factorize(x)[0])).astype("uint64")
        pass

    def _merge_user_item_feature_to_transactions(self, df_tra: pd.DataFrame) -> pd.DataFrame:
        """トランザクションログを受け取り、特徴量データをマージして返すメソッド.

        Parameters
        ----------
        df_tra : pd.DataFrame
            トランザクションログ

        Returns
        -------
        pd.DataFrame
            特徴量データをマージした、トランザクションログ
        """

        # トランザクションログ側のt_datを'週の最終日'に変換
        df_tra['t_dat'] = pd.to_datetime(df_tra['t_dat']).dt.to_period(
            'W').dt.to_timestamp(freq='W', how='end').dt.floor('D')
        # ユーザ特徴量をマージ
        print(df_tra.columns)
        print(self.item_features.columns)
        df_tra = df_tra.merge(self.user_features,
                              on='customer_id_short', how='left')
        # アイテム特徴量をマージ
        df_tra = df_tra.merge(
            self.item_features, on='article_id', how='left')
        # 降順(新しい順)で並び変え
        df_tra.sort_values(['t_dat', 'customer_id_short'],
                           inplace=True, ascending=False)
        # ラグ特徴量(アイテム)をマージ
        for target_column in Config.item_lag_feature_names_subcategory:
            # 対象サブカテゴリのラグ特徴量を取り出す
            lag_feature_df = self.item_lag_features[target_column]
            # t_datをobject型からdatetime型に
            lag_feature_df['t_dat'] = pd.to_datetime(lag_feature_df['t_dat'])

            # マージ
            df_tra = pd.merge(df_tra, lag_feature_df,
                              on=[target_column, 't_dat'], how='left'
                              )

        # ラグ特徴量(ユーザ)をマージ
        for target_column in Config.user_lag_feature_subcategory:
            # 対象サブカテゴリのラグ特徴量を取り出す
            lag_feature_df = self.user_lag_features[target_column]
            # t_datをobject型からdatetime型に
            lag_feature_df['t_dat'] = pd.to_datetime(lag_feature_df['t_dat'])

            # マージ
            df_tra = pd.merge(df_tra, lag_feature_df,
                              on=[target_column, 't_dat'], how='left'
                              )

        print('unique user of df_tra is {}'.format(
            len(df_tra['customer_id_short'].unique())
        ))

        return df_tra

    def _create_train_and_valid(self):
        N_ROWS = 1_000_000
        self.train = self.df.loc[self.df.t_dat <= (
            pd.to_datetime('2020-09-15') - self.date_minus)].iloc[:N_ROWS]
        print('unique user of self.train is {}'.format(
            len(self.train['customer_id_short'].unique())
        ))
        self.valid = self.df.loc[self.df.t_dat >= (
            pd.to_datetime('2020-09-16') - self.date_minus)]

        # delete transactions to save memory
        # del self.df

    def preprocessing(self):
        """前処理を実行するメソッド.
        """
        if Config.train_only_non_coldstart_user:
            self._extract_non_coldstart_user()

        self._create_df_1w_to4w()
        print(self._create_df_1w_to4w)
        self._load_feature_data()
        print(self._load_feature_data)
        self._preprocessing_user_feature()
        print(self._preprocessing_user_feature)
        self._create_train_and_valid()
        print(self._create_train_and_valid)

    def _create_purchased_dict(self):
        """過去の各ユーザのトランザクション情報をまとめる。
        この情報を元に、各ユーザに対するレコメンド「候補」n個を生成する。
        """

        def __create_purchased_dict(df_iw: pd.DataFrame) -> Tuple[Dict, List]:
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
        candidate_path: str = ''
        # 候補データのファイルパスを取得する。
        if Config.run_for_submittion:
            candidate_path = os.path.join(
                DRIVE_DIR, f'submission_csv/submission_{approach_name}.csv')
        elif Config.run_for_submittion == False:
            candidate_path = os.path.join(
                DRIVE_DIR,
                'val_results_{}_csv/val_{}.csv'.format(
                    self.val_week_id, approach_name)
            )

        # 読み込み(レコード：ユニークユーザ数、predictionカラムにレコメンド結果が入ってる。)
        candidates_df = pd.read_csv(candidate_path)
        # 'prediction'カラムを変換(str=>List[str]に)
        candidates_df['prediction'] = candidates_df['prediction'].apply(
            lambda x: x.split(' ')[:Config.num_negative_candidate])
        # explodeカラムで、[候補アイテムのリスト]をレコードに展開する！他のカラムの要素は複製される。
        candidates_df = candidates_df.explode('prediction')
        # 「候補」アイテムのカラムをRename
        candidates_df.rename(
            columns={'prediction': 'article_id'}, inplace=True)

        # customer_idカラムを落としておく
        print(candidates_df.columns)
        candidates_df.drop(columns='customer_id', inplace=True)

        # 後はコレをオリジナルとくっつければいいだけだけど...。
        return candidates_df

    def _create_positive_label_column(self):
        """ユーザ毎に# take only last 15 transactions。
        その後、各レコードにlabel=1(すなわち、購入あり)を付与する
        """
        self.train: pd.DataFrame

        if Config.train_only_non_coldstart_user == False:
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

            del self.train['rank']
            del self.train['rn']

        # トランザクションログの不要なカラム(=学習に使えない)を落とす
        self.train.drop(columns=['price', 'sales_channel_id'], inplace=True)

        self.train.sort_values(['t_dat', 'customer_id_short'], inplace=True)

        self.train['label'] = 1

        self.train.sort_values(['t_dat', 'customer_id_short'], inplace=True)
        print(f'length of positve train data is {len(self.train)}')

        # 検証用データに対しても同様(ユーザ毎に直近15件っていう制限はなくていいや)
        self.valid['label'] = 1

    def _create_negatives_for_train_using_lastDate_fromTrain(self):
        # 各ユーザに対して、学習データ期間の最終購入日を取得する
        # ＝＞重複を取り除く時に重要！
        last_dates = (
            self.train
            .groupby('customer_id_short')['t_dat']
            .max()
            .to_dict()
        )

        # 各ユーザに対して、「候補」アイテム(negative)をn個取得する。(transaction_dfっぽい形式になってる!)
        if Config.predict_candidate_way_name == None:
            self.negatives_df = self.__prepare_candidates_original(
                customers_id=self.train['customer_id_short'].unique(),
                n_candidates=Config.num_candidate_train)
        elif Config.predict_candidate_way_name == 'StaticPopularity_byfone':
            # Negativeサンプラーオブジェクトを使って、Negativeサンプル(DataFrame)を生成。
            self.negatives_df = NegativeSamplerStaticPopularity(
                dataset=self.dataset,
                transaction_train=self.df,
                val_week_id=self.val_week_id
            ).get_negative_record(unique_customer_ids=self.train['customer_id_short'].unique())

        else:
            self.negatives_df = self._load_candidate_from_other_recommendation()

        # negativeなレコードのt_datは、last_dates(使うのここだけ?)で穴埋めする。
        self.negatives_df['t_dat'] = self.negatives_df['customer_id_short'].map(
            last_dates)
        # データ型を一応変換しておく。
        self.negatives_df['article_id'] = self.negatives_df['article_id'].astype(
            'int')

        # negatives_dfのLabelカラムを0にする。
        self.negatives_df['label'] = 0
        print(f'negative_df columns is {self.negatives_df.columns}')

    def _create_negatives_for_valid_using_lastDate_fromTrain(self):
        """検証用データも同様の手順で、negativeを作る?
        """
        # 各ユーザに対して、学習データ期間の最終購入日を取得する。
        last_dates = (
            self.valid
            .groupby('customer_id_short')['t_dat']
            .max()
            .to_dict()
        )
        # 各ユーザに対して、「候補」アイテム(negative)をn個取得する。(transaction_dfっぽい形式になってる!)
        if Config.predict_candidate_way_name == None:
            self.negatives_df_valid = self.__prepare_candidates_original(
                customers_id=self.valid['customer_id_short'].unique(),
                n_candidates=Config.num_candidate_valid)

        elif Config.predict_candidate_way_name == 'StaticPopularity_byfone':
            # Negativeサンプラーオブジェクトを使って、Negativeサンプル(DataFrame)を生成。
            self.negatives_df_valid = NegativeSamplerStaticPopularity(
                dataset=self.dataset,
                transaction_train=self.df,
                val_week_id=self.val_week_id
            ).get_negative_record(unique_customer_ids=self.valid['customer_id_short'].unique())

        else:
            self.negatives_df_valid = self._load_candidate_from_other_recommendation()

        # negativeなレコードのt_datは、last_dates(使うのここだけ?)で穴埋めする。
        self.negatives_df_valid['t_dat'] = self.negatives_df_valid['customer_id_short'].map(
            last_dates)
        # データ型を一応変換しておく。
        self.negatives_df_valid['article_id'] = self.negatives_df_valid['article_id'].astype(
            'int')

        # negatives_dfのLabelカラムを0にする。
        self.negatives_df_valid['label'] = 0

        print(
            f'negative_df_valid columns is {self.negatives_df_valid.columns}')

    def _concat_train_and_negatives(self):
        """学習データのPositiveレコードとNegativeレコードを縦にくっつける。
        """
        print(f'length of positive in train is {len(self.train)}')
        print(f'length of negative in train is {len(self.negatives_df)}')

        # 縦にくっつける...重複ない??
        self.train = pd.concat([self.train, self.negatives_df], axis=0)

        # 重複を取り除く
        self.train.drop_duplicates(
            subset=['customer_id_short', 'article_id'],
            inplace=True)

        del self.negatives_df

    def _concat_valid_and_negatives(self):
        """同様に、検証データのPositiveレコードとNegativeレコードを縦にくっつける。
        """
        print(f'length of positive in train is {len(self.valid)}')
        print(f'length of negative in train is {len(self.negatives_df_valid)}')

        # 縦にくっつける...重複ない??
        self.valid = pd.concat([self.valid, self.negatives_df_valid], axis=0)

        # 重複を取り除く
        self.valid.drop_duplicates(
            subset=['customer_id_short', 'article_id'],
            inplace=True)

        del self.negatives_df_valid

    def _create_query_data_train(self):
        """
        学習データにクエリID列を持たせるメソッド。
        """
        # まずクエリIDを持たせる為に、customer_idでソートして、その中でt_datでソート
        self.train.sort_values(['customer_id_short', 't_dat'], inplace=True)

        self.train_baskets = self.train.groupby(['customer_id_short'])[
            'article_id'].count().values

    def _create_query_data_valid(self):
        # 検証用データに対しても同様に、クエリデータを生成する。
        self.valid.sort_values(['customer_id_short', 't_dat'], inplace=True)
        self.valid_baskets = self.valid.groupby(['customer_id_short'])[
            'article_id'].count().values

    def _save_feature_importance(self):
        # 特徴量重要度を保存する関数
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(20, 10))
        df_plt = pd.DataFrame({'feature_name': self.feature_names,
                               'feature_importance': self.ranker.feature_importances_}
                              )
        # 降順でソート
        df_plt.sort_values('feature_importance', ascending=False, inplace=True)
        # 保存
        df_plt.to_csv(os.path.join(
            DRIVE_DIR, f'feature/feature_importance_{self.val_week_id}.csv'))

        # 描画
        sns.barplot(x="feature_importance", y="feature_name", data=df_plt)
        plt.title('feature importance')
        # 保存
        plt.savefig(os.path.join(
            DRIVE_DIR, f'feature/feature_importance_{self.val_week_id}.png'))

        # Feature Importance上位50の特徴量を文字列のリストとして取得したい
        self.feature_names_highest50_feature_importance: List[str] = (
            df_plt.iloc[0:50, :]['feature_name'].values.tolist()
        )

    def fit(self):
        """Fit lightgbm ranker model
        """
        self._create_purchased_dict()
        self._create_positive_label_column()
        self._create_negatives_for_train_using_lastDate_fromTrain()
        self._concat_train_and_negatives()
        self._create_query_data_train()
        # ここで特徴量をマージ
        self.train = self._merge_user_item_feature_to_transactions(self.train)

        # 検証用データも同様に準備
        self._create_negatives_for_valid_using_lastDate_fromTrain()
        self._concat_valid_and_negatives()
        self._create_query_data_valid()
        # ここで特徴量をマージ
        self.valid = self._merge_user_item_feature_to_transactions(self.valid)

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
            ndcg_eval_at=[3, 5],
            class_weight='balanced'
        )
        # 一応、学習用データの最終日を取得し、出力
        last_date_train_data = self.train['t_dat'].max()
        init_date_train_data = self.train['t_dat'].min()
        print(f'dates of training data is..from{init_date_train_data}')
        print(f'dates of training data is..to{last_date_train_data}')
        print(f'len of train data is {len(self.train)}')

        # ラグ特徴量のカラム名を生成
        lag_feature_names = []
        for subcategory in Config.item_lag_feature_names_subcategory+Config.user_lag_feature_subcategory:
            for lag_kind in Config.item_lag_feature_names_kind:
                feature_name = f'{lag_kind}{subcategory}'
                lag_feature_names.append(feature_name)
        # 特徴量のカラム名を保存
        self.feature_names = (
            Config.item_basic_feature_names
            + Config.item_numerical_feature_names
            + Config.item_one_hot_encoding_feature_names
            + Config.user_numerical_feature_names
            + Config.hidden_variable_feature_names
            + lag_feature_names
        )
        # 特徴量とターゲットを分割
        X_train = self.train[self.feature_names]
        y_train = self.train['label']
        print(f'length of X_valid is {len(X_train.columns)}')
        X_valid = self.valid[self.feature_names]
        y_valid = self.valid['label']
        print(f'length of X_valid is {len(X_valid.columns)}')
        # Categorical Featureの指定
        self.categorical_feature_names: List[str] = list(
            X_train.select_dtypes(include=int).columns)
        print(
            f'categorical feature names are {self.categorical_feature_names}')
        # 学習
        self.ranker = self.ranker.fit(
            X=X_train,
            y=y_train,
            group=self.train_baskets,
            eval_set=[(X_valid, y_valid)],
            eval_group=[list(self.valid_baskets)],
            eval_metric="ndcg",
            feature_name=self.feature_names,
            categorical_feature=self.categorical_feature_names
        )

        # Feature Importanceを取得
        self._save_feature_importance()

        # 学習済みモデルの保存
        if Config.save_trained_model:
            filepath = os.path.join(DRIVE_DIR, 'trained_lgbm_model.pkl')
            pickle.dump(obj=self.ranker, file=open(filepath, 'wb'))

        # Saving memoryの為、学習用と検証用のデータセットを削除
        del self.train, self.valid

    def _prepare_candidate(self):
        """予測の準備をするメソッド。
        具体的には、
        - sample_submissionを読み込んでおく.
        - 予測する各ユニークユーザに対して、Candidateを用意。
        - 作成されたDataFrame(レコード数= len(unique ser) * n_candidate)に対して、
        アイテム特徴量とユーザ特徴量をマージ
        """
        self.sample_sub = self.dataset.df_sub[[
            'customer_id_short', 'customer_id']].copy()

        if Config.predict_only_non_coldstart_user:
            self.sample_sub = self.sample_sub.merge(
                self.user_activity_df[[
                    'customer_id_short', 'cold_start_status']],
                on='customer_id_short', how='left'
            )
            # non_cold_startなユーザのみを予測対象とする。
            self.sample_sub = self.sample_sub.loc[self.sample_sub['cold_start_status']
                                                  == 'non_cold_start']
            # cold_startなユーザは、静的なレコメンドを実行(Chris? Time decay?)
            self.sample_sub_cold_start_user = self.sample_sub.loc[
                self.sample_sub['cold_start_status'] == 'cold_start']

        self.candidates = pd.DataFrame()
        # レコメンド候補を用意
        if Config.predict_candidate_way_name == None:
            # NoneだったらオリジナルのCandidate
            self.candidates = self.__prepare_candidates_original(
                self.sample_sub['customer_id_short'].unique(), Config.num_candidate_predict)
            print(f'length of prediction candidates is {len(self.candidates)}')

        elif Config.predict_candidate_way_name == 'StaticPopularity_byfone':
            # Negativeサンプラーオブジェクトを使って、Negativeサンプル(DataFrame)を生成。
            self.candidates = NegativeSamplerStaticPopularity(
                dataset=self.dataset,
                transaction_train=self.df,
                val_week_id=self.val_week_id
            ).get_prediction_candidates(unique_customer_ids=self.sample_sub['customer_id_short'].unique())
        else:
            self.candidates = self._load_candidate_from_other_recommendation()

        print(f'columns of candidates are {self.candidates.columns}')
        # article_idのデータ型をindに
        self.candidates['article_id'] = self.candidates['article_id'].astype(
            'int')

    def _predict_using_batches(self):
        """実際に予測を実行するメソッド。メモリの関係からbatch_size数ずつ入力していく.
        Predict using batches, otherwise doesn't fit into memory.
        """
        def _merge_featureData_to_candidate(candidates_batch: pd.DataFrame):
            """バッチサイズ分のCandidatesに対して、特徴量データをマージする関数.
            """

            # Candidatesのt_datカラムを生成&検証週の最終日に
            last_date_in_test_week = pd.to_datetime(
                '2020-09-27') - timedelta(days=7 * (105-self.val_week_id))
            candidates_batch['t_dat'] = pd.to_datetime(last_date_in_test_week)

            # ユーザ特徴量＆アイテム特徴量を結合
            candidates_batch = (
                candidates_batch
                .merge(self.user_features, on=('customer_id_short'), how='left')
                .merge(self.item_features, on=('article_id'), how='left')
            )

            # ラグ特徴量（アイテム）をマージ
            for target_column in Config.item_lag_feature_names_subcategory:
                # 対象サブカテゴリのラグ特徴量を取り出す
                lag_feature_df = self.item_lag_features[target_column]
                # マージ
                candidates_batch = pd.merge(candidates_batch, lag_feature_df,
                                            on=[target_column, 't_dat'], how='left'
                                            )

            # ラグ特徴量（ユーザ）をマージ
            for target_column in Config.user_lag_feature_subcategory:
                # 対象サブカテゴリのラグ特徴量を取り出す
                lag_feature_df = self.user_lag_features[target_column]
                # マージ
                candidates_batch = pd.merge(candidates_batch, lag_feature_df,
                                            on=[target_column, 't_dat'], how='left'
                                            )

            return candidates_batch

        # 予測値のリストをInitialize
        self.preds = []
        # batchサイズ分ずつ、予測していく.
        batch_size = 1_000_000
        for bucket in tqdm(range(0, len(self.candidates), batch_size)):
            print(
                f'predict process with index from {bucket} to {bucket+batch_size}')
            # candidateからバッチサイズ分抽出
            candidates_batch = self.candidates.iloc[bucket:(
                bucket + batch_size), :]
            # 特徴量データを結合する
            candidates_batch = _merge_featureData_to_candidate(
                candidates_batch)
            # 特徴量を用意。
            X_pred = candidates_batch[self.feature_names]
            # モデルに特徴量を入力して、出力値を取得
            outputs = self.ranker.predict(X=X_pred)
            # -> (ユニークユーザ × num_candidate_predict)個のうち、バッチサイズ分のトランザクションの発生確率を返す?
            # 予測値のリストにndarrayを追加
            self.preds.append(outputs)

    def _create_recommendation_by_ranking(self):
        # 各バッチ毎の予測結果(ndarrayのリスト)を縦に結合(1つのndarrayに)
        # ->(ユニークユーザ × num_candidate_predict)個のレコードの発生確率y
        self.preds = np.concatenate(self.preds)
        print(f'type of prediction results is {type(self.preds)}')
        print(f'length of prediction results is {self.preds.shape}')
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

        # カラム名を修正
        self.preds = self.sample_sub[['customer_id_short', 'customer_id']].merge(
            self.preds
            .reset_index()
            .rename(columns={'article_id': 'prediction'}), how='left')

    def _prepare_submission(self):
        recommend_result = pd.DataFrame()
        # モデルでレコメンドしきれていないユーザ(cold startユーザ)用のレコメンド
        if Config.approach_name_for_coldstart_user == 'time_decaying':
            # レコメンド結果を読み込み
            filepath = os.path.join(
                DRIVE_DIR, 'submission_csv/submission_exponentialDecay.csv')
            recommend_result = pd.read_csv(filepath)

        # customer_id_shortカラムがない場合の対処
        if 'customer_id_short' not in recommend_result.columns:
            recommend_result['customer_id_short'] = recommend_result["customer_id"].apply(
                lambda s: int(s[-16:], 16)).astype("uint64")

        # レコメンド内容をcoldstart ユーザに対してマージ
        self.sample_sub_cold_start_user = pd.merge(
            self.sample_sub_cold_start_user,
            recommend_result, on='customer_id_short', how='left'
        )

        # non_cold_startユーザの結果とcold_startユーザの結果をConcat
        self.preds = pd.concat(objs=[
            self.preds[['customer_id', 'customer_id_short', 'prediction']],
            self.sample_sub_cold_start_user[[
                'customer_id', 'customer_id_short', 'prediction']]
        ],
            axis=0  # 縦方向の連結
        )

    def create_reccomendation(self) -> pd.DataFrame:
        self._prepare_candidate()
        self._predict_using_batches()
        self._create_recommendation_by_ranking()
        self._prepare_submission()

        return self.preds[['customer_id', 'customer_id_short', 'prediction']]


if __name__ == '__main__':
    pass
