from datetime import timedelta
from typing import Dict, List, Tuple

from config import Config
import pandas as pd
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm
from models.Byfone_appraoch_moreSpeedy import ByfoneModel


class NegativeSamplerStaticPopularity:
    def __init__(self, dataset: DataSet, transaction_train: pd.DataFrame, val_week_id: int) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.df_t = transaction_train
        self.val_week_id = val_week_id

    def get_negative_record(self, unique_customer_ids):
        self.negatives_df:pd.DataFrame
        self.unique_customer_ids = unique_customer_ids
        self.n_negative = Config.num_candidate_train

        self._get_quotient_each_item()
        self._set_weights_of_sampling()
        self._create_negative_sampler()
        return self.negatives_df

    def _get_quotient_each_item(self):
        # まずquotientを計算
        byfone_model = ByfoneModel(self.df_t,
                                   self.dataset,
                                   val_week_id=self.val_week_id)
        byfone_model.preprocessing()
        # quotientが計算されたトランザクションログを取得
        self.df_t = byfone_model.transaction_train

        # 各アイテム毎のquotientの合計値を集計する
        self.quotient_each_item = self.df_t.groupby('article_id')['quotient'].sum().reset_index()
        # -> 各レコード:ユニークユーザ、カラム：article_idとquotient


    def _set_weights_of_sampling(self):
        """各ユニークアイテムの特徴量(=ex. 人気度)をベースにサンプラーの重みを設定
        """

        # 各レコード(アイテム)が抽出される確率(=重み)として、各アイテム毎のquotientを使用
        self.weights_sampler: List[float] = self.quotient_each_item['quotient'].values.tolist()
        # 総和が1.0になるように調整
        sum_quotient_all_item = sum(self.weights_sampler)
        self.weights_sampler = [(w/sum_quotient_all_item) for w in self.weights_sampler]


    def _create_negative_sampler(self):

        # 結果のDictをInitialize:Dict[ユーザid, 「候補」リスト]
        self.prediction_dict = {}
        # 各ユーザ毎に繰り返し処理で、Negativeサンプルを付与していく。
        for i, cust_id in tqdm(enumerate(self.unique_customer_ids)):
            # 対象ユーザのNegativeサンプルを生成
            negative_sample = self.quotient_each_item['article_id'].sample(
                n=self.n_negative,
                weights=self.weights_sampler, # 各行が抽出される確率リスト
                axis=0, # 行を抽出
                replace=False, # 重複無し
            )
            # 対象ユーザのNegativeサンプルをdictに格納
            self.prediction_dict[cust_id] = negative_sample

        # ＝＞user_idと「候補」をそれぞれリストで取得。
        k = list(map(lambda x: x[0], self.prediction_dict.items()))
        v = list(map(lambda x: x[1], self.prediction_dict.items()))

        # DataFrameとして保存.(user_id, [候補アイテムのリスト])
        self.negatives_df = pd.DataFrame({'customer_id_short': k, 'negatives': v})
        # explodeメソッドで、各ユーザの[negative sampleのリスト]をレコードに展開する！他のカラムの要素は複製される。
        self.negatives_df = self.negatives_df.explode('negatives')
        # 「候補」アイテムのカラムをRename
        self.negatives_df.rename(columns={'negatives': 'article_id'}, inplace=True)

        self.negatives_df = self.negatives_df[['customer_id_short', 'article_id']]

    def get_prediction_candidates(self, unique_customer_ids):
        self.negatives_df:pd.DataFrame
        self.unique_customer_ids = unique_customer_ids
        self.n_negative = Config.num_candidate_predict

    def _create_byfone_recommendation(self):
        byfone_model = ByfoneModel(self.df_t,
                                   self.dataset,
                                   val_week_id=self.val_week_id)
        byfone_model.preprocessing()
        # レコメンド結果を取得(str型になってるので変換が必要)
        self.df_candidates = byfone_model.create_reccomendation()
        # str=> strのlistへ
        self.df_candidates['prediction'] = self.df_candidates['prediction'].apply(lambda x: x.split(' '))
        # strのList=>intのListへ
        self.df_candidates['prediction'] = self.df_candidates['prediction'].apply(
            lambda x: [int(article_id) for article_id in x]
            )
        print(f'check data type of candidates: {self.df_candidates['prediction'][0]}')
        # explodeメソッドで、各ユーザの[predictionのリスト]をレコードに展開する！他のカラムの要素は複製される。
        self.df_candidates = self.df_candidates.explode('prediction')
        # 「候補」アイテムのカラムをRename
        self.df_candidates.rename(columns={'prediction': 'article_id'}, inplace=True)

        # 最終的には2つのカラム
        self.df_candidates = self.df_candidates[['customer_id_short', 'article_id']]

if __name__ == '__main__':
    pass
