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
        date_minus = timedelta(days=7*(105 - self.val_week_id))
        # トランザクション
        df = self.dataset.df.copy()
        df['article_id'] = df['article_id'].astype('int')
        self.df = df

        self.df_1w = df[df['t_dat'] >= (
            pd.to_datetime('2020-09-15') - date_minus)].copy()
        self.df_2w = df[df['t_dat'] >= (
            pd.to_datetime('2020-09-07') - date_minus)].copy()
        self.df_3w = df[df['t_dat'] >= (
            pd.to_datetime('2020-08-31') - date_minus)].copy()
        self.df_4w = df[df['t_dat'] >= (
            pd.to_datetime('2020-08-24') - date_minus)].copy()

    def _load_feature_data(self):
        self.item_features = pd.read_parquet(os.path.join(
            DRIVE_DIR, 'input/item_features.parquet')).reset_index()
        self.user_features = pd.read_parquet(os.path.join(
            DRIVE_DIR, 'input/user_features.parquet')).reset_index()
        # customer_id_shortカラムを作成
        self.user_features["customer_id_short"] = self.user_features["customer_id"].apply(lambda s: int(s[-16:], 16)).astype("uint64")

    def _preprocessing_user_feature(self):
        self.user_features[['club_member_status', 'fashion_news_frequency']]= (
                   self.user_features[['club_member_status',
                       'fashion_news_frequency']]
                   .apply(lambda x: pd.factorize(x)[0])).astype('int8')

    def _merge_user_item_feature_to_transactions(self):
        self.df= self.df.merge(self.user_features, on = ('customer_id_short'))
        self.df= self.df.merge(self.item_features, on = ('article_id'))
        self.df.sort_values(['t_dat', 'customer_id'], inplace=True)

    def _create_train_and_valid(self):
        self.train = self.df.loc[self.df.t_dat <= (pd.to_datetime('2020-09-15') - timedelta(7))]
        self.valid = self.df.loc[self.df.t_dat >= (pd.to_datetime('2020-09-16') - timedelta(7))]

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

        def __create_purchased_dict(df_iw: pd.DataFrame):
            purchase_dict_iw= {}
            for i, x in enumerate(zip(df_iw['customer_id_short'], df_iw['article_id'])):
                cust_id, art_id= x
                if cust_id not in purchase_dict_iw:
                    purchase_dict_iw[cust_id]= {}

                if art_id not in purchase_dict_iw[cust_id]:
                    purchase_dict_iw[cust_id][art_id]= 0

                purchase_dict_iw[cust_id][art_id] += 1

            dummy_list_iw= list((df_iw['article_id'].value_counts()).index)[:12]

            return purchase_dict_iw, dummy_list_iw

        self.purchase_dict_4w, self.dummy_list_4w= __create_purchased_dict(self.df_4w)
        self.purchase_dict_3w, self.dummy_list_3w= __create_purchased_dict(self.df_3w)
        self.purchase_dict_2w, self.dummy_list_2w= __create_purchased_dict(self.df_2w)
        self.purchase_dict_1w, self.dummy_list_1w= __create_purchased_dict(self.df_1w)

    def _prepare_candidates(self, customers_id, n_candidates: int = 12):
        """
        df - basically, dataframe with customers(customers should be unique)
        """
        prediction_dict= {}
        dummy_list= list((self.df_2w['article_id'].value_counts()).index)[:n_candidates]

        for i, cust_id in tqdm(enumerate(customers_id)):
            # comment this for validation
            if cust_id in self.purchase_dict_1w:
                l= sorted((self.purchase_dict_1w[cust_id]).items(), key=lambda x: x[1], reverse=True)
                l=[y[0] for y in l]
                if len(l) > n_candidates:
                    s=l[:n_candidates]
                else:
                    s=l+self.dummy_list_1w[:(n_candidates-len(l))]
            elif cust_id in self.purchase_dict_2w:
                l=sorted((self.purchase_dict_2w[cust_id]).items(
                ), key=lambda x: x[1], reverse=True)
                l=[y[0] for y in l]
                if len(l) > n_candidates:
                    s=l[:n_candidates]
                else:
                    s=l+self.dummy_list_2w[:(n_candidates-len(l))]
            elif cust_id in self.purchase_dict_3w:
                l=sorted((self.purchase_dict_3w[cust_id]).items(
                ), key=lambda x: x[1], reverse=True)
                l=[y[0] for y in l]
                if len(l) > n_candidates:
                    s=l[:n_candidates]
                else:
                    s=l+self.dummy_list_3w[:(n_candidates-len(l))]
            elif cust_id in self.purchase_dict_4w:
                l=sorted((self.purchase_dict_4w[cust_id]).items(
                ), key=lambda x: x[1], reverse=True)
                l=[y[0] for y in l]
                if len(l) > n_candidates:
                    s=l[:n_candidates]
                else:
                    s=l+self.dummy_list_4w[:(n_candidates-len(l))]
            else:
                s=dummy_list
                prediction_dict[cust_id]=s

        k=list(map(lambda x: x[0], prediction_dict.items()))
        v=list(map(lambda x: x[1], prediction_dict.items()))
        negatives_df=pd.DataFrame({'customer_id_short': k, 'negatives': v})
        negatives_df=(
            negatives_df
            .explode('negatives')
            .rename(columns={'negatives': 'article_id'})
            )
        return negatives_df

    def _create_rank_column(self):
        # take only last 15 transactions
        self.train['rank'] = range(len(self.train))
        self.train = (
            self.train
            .assign(
                rn = self.train.groupby(['customer_id_short'])['rank']
                        .rank(method='first', ascending=False))
            .query("rn <= 15")
            .drop(columns = ['price', 'sales_channel_id'])
            .sort_values(['t_dat', 'customer_id_short'])
        )
        self.train['label'] = 1

        del self.train['rank']
        del self.train['rn']

        self.valid.sort_values(['t_dat', 'customer_id_short'], inplace = True)
    
    def _append_negatives_to_positives_using_lastDate_fromTrain(self):
        last_dates = (
            self.train
            .groupby('customer_id_short')['t_dat']
            .max()
            .to_dict()
        )

        self.negatives = self._prepare_candidates(customers_id=self.train['customer_id_short'].unique(), n_candidates=15)
        self.negatives['t_dat'] = self.negatives['customer_id_short'].map(last_dates)
        self.negatives['article_id'] = self.negatives['article_id'].astype('int')
        self.negatives = (
            self.negatives
            .merge(self.user_features, on = ('customer_id_short'))
            .merge(self.item_features, on = ('article_id'))
        )
        self.negatives['label'] = 0

    def _merge_train_and_negatives(self):
        self.train = pd.concat([self.train, self.negatives])
        self.train.sort_values(['customer_id_short', 't_dat'], inplace = True)
        self.train_baskets = self.train.groupby(['customer_id'])['article_id'].count().values

    def fit(self):
        """Fit lightgbm ranker model
        """
        self._create_purchased_dict()
        self._create_rank_column()
        self._append_negatives_to_positives_using_lastDate_fromTrain()
        self._merge_train_and_negatives()

        self.ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="dart",
            max_depth=7,
            n_estimators=300,
            importance_type='gain',
            verbose=10, 
        )

        X_train = self.train.drop(columns=['t_dat', 'customer_id', 'customer_id_short', 'article_id', 'label'])
        print(X_train.columns)
        y_train = self.train.pop('label')
        self.ranker = self.ranker.fit(
            X=X_train,
            y=y_train,
            group=self.train_baskets
        )


    def _prepare_prediction(self):
        self.sample_sub = self.dataset.cid

        self.candidates = self._prepare_candidates(self.sample_sub['customer_id_short'].unique(), 12)
        self.candidates['article_id'] = self.candidates['article_id'].astype('int')
        self.candidates = (
            self.candidates
            .merge(self.user_features, on = ('customer_id_short'))
            .merge(self.item_features, on = ('article_id'))
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
            X_pred = self.candidates.iloc[bucket: bucket+batch_size].drop(columns = ['customer_id_short', 'customer_id', 'article_id'])
            print(X_pred.columns)
            # モデルに特徴量を入力して、出力値を取得
            outputs = self.ranker.predict(X=X_pred)
            # 予測値のリストに追加
            self.preds.append(outputs)

    def _prepare_submission(self):
        self.preds = np.concatenate(self.preds)
        
        self.candidates['preds'] = self.preds
        self.preds = self.candidates[['customer_id_short', 'article_id', 'preds']]
        self.preds.sort_values(['customer_id_short', 'preds'], ascending=False, inplace = True)
        self.preds = (
            self.preds.groupby('customer_id_short')[['article_id']].aggregate(lambda x: x.tolist())
        )
        # 提出用にレコメンドアイテムの体裁を整える。
        self.preds['article_id'] = self.preds['article_id'].apply(lambda x: ' '.join(['0'+str(k) for k in x]))

        # モデルでレコメンドしきれていないユーザを補完
        self.preds = self.sample_sub[['customer_id_short']].merge(
            self.preds
            .reset_index()
            .rename(columns = {'article_id': 'prediction'}), how = 'left')
        self.preds['prediction'].fillna(' '.join(['0'+str(art) for art in self.dummy_list_2w]), inplace = True)


    def create_reccomendation(self)->pd.DataFrame:
        self._prepare_prediction()
        self._predict_using_batches()
        self._prepare_submission()

        return self.preds


if __name__ == '__main__':
    pass
