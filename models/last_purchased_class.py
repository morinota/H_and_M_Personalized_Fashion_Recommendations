
from typing import List, Tuple
import pandas as pd
from sympy import arg
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
from collections import defaultdict
from config import Config
from datetime import timedelta


class LastPurchasedItrems:

    def __init__(self, transaction_train: pd.DataFrame, dataset: DataSet, val_week_id: int, k=12) -> None:
        # インスタンス変数(属性の初期化)
        self.dataset = dataset
        self.df = transaction_train
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.hyper_params = {}
        self.val_week_id = val_week_id
        self.k = k

    def __get_last_purchased_date(self):
        # 各ユーザの、学習期間内における、最終購入日を取得？
        self.last_purchase = self.df.groupby('customer_id_short')["t_dat"].max(
        ).reset_index().rename(columns={'t_dat': 'last_buy_date'})
        # ->カラム["customer_id_short" ,"last_buy_date"]

        # # 各ユーザの最終購入日を学習用のtransactionデータにマージ
        self.df = pd.merge(
            self.df, self.last_purchase, on='customer_id_short', how='left')

    def __get_dif_date_from_last_purchased(self):
        # 「各transactionの購入日と、最終購入日との差」を示す"dif_date"カラムを作成
        self.df['dif_date'] = (
            self.df["last_buy_date"] - self.df["t_dat"])

        # pd.Series.dt属性で、日付データの粒度を変換する(年月日＝＞日へ)
        self.df['dif_date'] = self.df['dif_date'].dt.days

    def __extract_within_2weeks_and_sort_descending(self):
        # 「最終購入日との差が2週間以内のもののみを抽出 & 購入日の降順に並び変え.
        self.df = self.df.loc[self.df["dif_date"] < 14]\
            .sort_values(['t_dat'], ascending=False)
        # アイテム×ユーザの重複を取り除く(drop_duplicatesは、指定したカラムの全てが重複した行を削除)
        self.df = self.df.drop_duplicates(
            ['customer_id_short', 'article_id'])

    def __summarize_each_user_last_purchased_item(self):
        # 各ユーザ毎に、最終購入日から二週間以内に購入したアイテム達をまとめる。
        # &(リスト=>strへ変換)&(設定されたindexをカラム化)
        self.df_sub_last_purchased_items = self.df.groupby(
            'customer_id_short')['article_id'].apply(iter_to_str).reset_index()
        # ->カラム=[customer_id_short, article_id]、レコードは各ユーザ。
        # article_idカラムの要素は「最終購入日から二週間以内に購入したアイテム達」を繋げたstr.

    def __complement_lack_of_user(self):

        # ユーザの不足分(トランザクションデータに含まれていないユーザ)を追加する
        df_sub = self.dataset.df_sub[[
            'customer_id', 'customer_id_short']].copy()
        self.df_sub_last_purchased_items = pd.merge(self.df_sub_last_purchased_items[['customer_id_short', 'article_id']],
                                                    df_sub,
                                                    on='customer_id_short',
                                                    how='right'
                                                    )
        # 補完したユーザのlast_purchased_itemの欠損値を''で埋める
        self.df_sub_last_purchased_items['article_id'].fillna('', inplace=True)
        # レコメンド結果のカラム名をpredictionに
        self.df_sub_last_purchased_items.rename(
            columns={'article_id': 'prediction'}, inplace=True)

    def _create_recommend_candidates_based_on_last_purchased_items(self) -> pd.DataFrame:
        self.__get_last_purchased_date()
        self.__get_dif_date_from_last_purchased()
        self.__extract_within_2weeks_and_sort_descending()
        self.__summarize_each_user_last_purchased_item()
        self.__complement_lack_of_user()

        # 最終的なカラムは3つのみにしておく
        self.df_sub_last_purchased_items = self.df_sub_last_purchased_items[[
            'customer_id_short', 'customer_id', 'prediction']]
        return self.df_sub_last_purchased_items

    '''
    以下、other_colors_of_purchsed_item
    -------------------------------------
    '''

    def __summarize_sales_count_each_item(self):
        # 設定した期間のtransactionデータに対して、各アイテムの購入回数を集計?
        self.df_item_counts_trainTerm: pd.DataFrame = self.df.groupby(
            ['article_id'])["t_dat"].count().reset_index()
        # ->カラム["article_id"k, "t_dat"(=各アイテムの購入回数)]
        # カラム名をt_datからcount_termに変更
        self.df_item_counts_trainTerm.rename(
            columns={'t_dat': 'count_term'}, inplace=True
        )

    def __merge_item_meta_with_sales_count_each_item(self):
        # 「設定した期間内における各アイテムの購入回数」とアイテムメタデータをマージ。
        self.article_df: pd.DataFrame = pd.merge(
            self.df_item_counts_trainTerm, self.dataset.dfi, on='article_id', how='right'
        )

    def __sort_by_count_term_descending(self):
        # count_term(設定した期間内における各アイテムの購入回数)が多い順にソート &
        self.popular_item_ranking = self.article_df.sort_values(
            'count_term', ascending=False)
        # count_termが1以上のアイテムのみ抽出。
        self.popular_item_ranking.query('count_term > 0', inplace=True)

    def __get_one_similar_item_with_each_item(self):
        """直近一週間に購入された各商品に対して、類似商品のarticle_idを1つだけ取得
        """
        # dictを詳しく定義できる関数?(collectionsモジュール)
        self.map_to_col = defaultdict(list)
        # 各アイテム毎に繰り返し処理：
        for article_id in self.popular_item_ranking['article_id'].tolist():
            # product_codeカラムは、article_id(9桁)を粗くしたモノ(上6桁)。(＝＞同種のアイテムの違う色とかっぽい！)
            alike_product_code_mask = self.popular_item_ranking['product_code'] == (
                int(article_id) // 1000)  # 「a // b」aをbで割った商の整数値
            # 直近一週間に購入された各商品に対して、類似商品のarticle_idを1つだけ、List型として、Dictに保存
            self.map_to_col[article_id] = list(filter(
                lambda x: x != article_id, self.popular_item_ranking[alike_product_code_mask]["article_id"].tolist()))[:1]

    def __recommend_similar_item_to_each_user(self):
        """# 各ユーザのLast_purchased_itemを元に、それらの類似アイテムをレコメンドする。
        """
        # 各ユーザのLast_purchased_itemを元に、それらの類似アイテムをレコメンドする。
        def __map_to_variation(s):
            '''
            「各ユーザのlast_purchase」の各商品のリストを入力とし、それらの類似商品を返す関数。apply()メソッド用。
            '''

            def f(item): return iter_to_str(self.map_to_col[int(item)])
            return ' '.join(map(f, s.split()))

        # 上記関数の処理をapplyで適用。other_colorsのレコメンド結果を生成。
        self.df_sub_other_colors = self.dataset.df_sub[[
            'customer_id', 'customer_id_short']].copy()
        self.df_sub_other_colors['prediction'] = self.df_sub_last_purchased_items['prediction'].fillna(
            '').apply(__map_to_variation)
        # 欠損値は''で補完
        self.df_sub_other_colors['prediction'].fillna('', inplace=True)

    def _create_recommend_candidates_based_on_other_colors_of_purchased_item(self):
        self.__summarize_sales_count_each_item()
        self.__merge_item_meta_with_sales_count_each_item()
        self.__sort_by_count_term_descending()
        # ここまでで学習期間内の、人気アイテムランキング的なものができた。
        self.__get_one_similar_item_with_each_item()
        self.__recommend_similar_item_to_each_user()

        # 最終的なカラムは3つのみにしておく
        self.df_sub_other_colors = self.df_sub_other_colors[[
            'customer_id_short', 'customer_id', 'prediction']]
        return self.df_sub_other_colors

    '''
    以下、popular_items_for_each_group
    -------------------------------------------------------------------------
    '''

    def __add_grouping_to_transaction_log(self):
        # group_dfをtransactionデータにマージする事で、各transactionにグルーピングを付与する。
        self.df = self.df.merge(
            self.grouping_df, on='customer_id_short', how='left')

    def __salescount_each_grouping(self):
        # グループ毎に「設定した期間内における各アイテムの購入回数」をカウントする
        self.item_salescount_each_group = self.df.groupby(['group', 'article_id'])[
            "t_dat"].count().reset_index()
        # groupbyの後のリセットインデックス大事だわ！
        # -> カラム=[group, article_id, t_dat]、レコードは各アイテム。

    def __get_popular_items_at_k_with_each_group(self):
        """各グループで、「設定した期間内における各アイテムの購入回数」の多い上位12個をリストで取得
        """
        k = Config.num_recommend_item
        self.items = defaultdict(str)
        # 各グループ毎に繰り返し処理：
        for group in self.item_salescount_each_group["group"].unique():
            # 各グループで、「設定した期間内における各アイテムの購入回数」の多い上位12個をリストで取得
            # List＝＞strに変換
            # dictのvalueとして保存。keyはグループを示すカテゴリ変数。
            self.items[group] = iter_to_str(self.item_salescount_each_group.loc[self.item_salescount_each_group["group"] == group].sort_values(
                't_dat', ascending=False)["article_id"].tolist()[:12])

    def __add_popular_items_to_each_user(self):
        self.df_sub_popular_items_each_group = self.dataset.df_sub[[
            'customer_id', 'customer_id_short']].copy()
        # 各ユーザに対して、「グループ別の人気アイテム」を取得。レコメンド結果を保存
        self.df_sub_popular_items_each_group['prediction'] = self.grouping_df["group"].map(
            self.items)
        # 欠損値は''で補完
        self.df_sub_other_colors['prediction'].fillna('', inplace=True)

    def _create_recommend_candidates_based_on_popular_items_for_each_group(self, grouping_df):
        self.grouping_df = grouping_df
        self.__add_grouping_to_transaction_log()
        self.__salescount_each_grouping()
        self.__get_popular_items_at_k_with_each_group()
        self.__add_popular_items_to_each_user()

        # 最終的なカラムは3つのみにしておく
        self.df_sub_popular_items_each_group = self.df_sub_popular_items_each_group[[
            'customer_id_short', 'customer_id', 'prediction']]
        return self.df_sub_popular_items_each_group

    '''
    3つのCandidatesを結合する。
    ------------------------------------------------------------------------------------
    '''

    def _generate_set_recent_sold(self):
        """ 直近の一定期間内(ex. 直近12日)に一度でも購入された、アイテムidの集合を生成
        """
        interval_days = 11
        self.last_date = self.df['t_dat'].max()
        self.init_date = self.last_date - timedelta(days=interval_days)
        # 直近の一定期間内のトランザクションログを抽出
        df_sold_more_than_onetime_recently = self.df.loc[
            (self.df['t_dat'] >= self.init_date) &
            (self.df['t_dat'] <= self.last_date)
        ]
        # 直近の一定期間内に一度でも購入された、アイテムidの集合を生成
        self.sold_set = set(
            df_sold_more_than_onetime_recently['article_id'].to_list())

    def _merge_three_recommendation_candidates(self):
        """各レコメンド手法の予測結果のDataFrameをマージする。
        """
        self.df_sub_unioned = self.dataset.df_sub[[
            'customer_id', 'customer_id_short']].copy()
        print(len(self.df_sub_unioned))
        print(len(self.df_sub_last_purchased_items))
        print(len(self.df_sub_other_colors))
        print(len(self.df_sub_popular_items_each_group))

        self.df_sub_unioned = pd.merge(
            left=self.df_sub_unioned,
            right=self.df_sub_last_purchased_items[['customer_id_short', 'prediction']].rename(
                columns={'prediction': 'last_purchase'}),
            on='customer_id_short',
            how='left'
        )
        self.df_sub_unioned = pd.merge(
            left=self.df_sub_unioned,
            right=self.df_sub_other_colors[['customer_id_short', 'prediction']].rename(
                columns={'prediction': 'other_colors'}),
            on='customer_id_short',
            how='left'
        )
        self.df_sub_unioned = pd.merge(
            left=self.df_sub_unioned,
            right=self.df_sub_popular_items_each_group[['customer_id_short', 'prediction']].rename(
                columns={'prediction': 'popular_items'}),
            on='customer_id_short',
            how='left'
        )

        print(self.df_sub_unioned.isna().sum())

    def _union_three_recommendation_candidates(self):
        """各レコメンド手法の予測結果を合体させる。
        """
        # innor function 1
        def blend(dt, w=[], k=12):
            '''
            dt:datatimeの事.
            '''
            if len(w) == 0:
                w = [1] * (len(dt))
            preds = []
            for i in range(len(w)):
                
                preds.append(dt[i].split())
            res = {}
            for i in range(len(preds)):
                if w[i] < 0:
                    continue
                for n, v in enumerate(preds[i]):
                    if v in res:
                        res[v] += (w[i] / (n + 1))
                    else:
                        res[v] = (w[i] / (n + 1))
            res = list(
                dict(sorted(res.items(), key=lambda item: -item[1])).keys())
            return ' '.join(res[:k])

        # innor function 2
        def prune(pred, ok_set, k=12):
            pred = pred.split()
            post = []
            for item in pred:
                if int(item) in ok_set and not item in post:
                    post.append(item)
            return " ".join(post[:k])

        # まずは、blend()で合体させる。
        self.df_sub_unioned[[
            'last_purchase', 'other_colors', 'popular_items']]
        self.df_sub_unioned['prediction'] = self.df_sub_unioned[[
            'last_purchase', 'other_colors', 'popular_items']].apply(blend,
                                                                     w=[100, 10, 1], axis=1, k=32)

        # 次に、prune()で切り取る。
        self.df_sub_unioned['prediction'].apply(
            prune, ok_set=self.df_sub_unioned)

    def create_recommendation(self, grouping_df):
        # まず3種類のレコメンド結果を生成
        self._create_recommend_candidates_based_on_last_purchased_items()
        print(self._create_recommend_candidates_based_on_last_purchased_items)
        self._create_recommend_candidates_based_on_other_colors_of_purchased_item()
        print(self._create_recommend_candidates_based_on_other_colors_of_purchased_item)
        self._create_recommend_candidates_based_on_popular_items_for_each_group(
            grouping_df)
        print(self._create_recommend_candidates_based_on_popular_items_for_each_group)

        # 以下、ユニオンする処理
        self._generate_set_recent_sold()
        self._merge_three_recommendation_candidates()
        self._union_three_recommendation_candidates()

        # 最終的なカラムは3つのみにしておく
        self.df_sub_unioned = self.df_sub_unioned[[
            'customer_id_short', 'customer_id', 'prediction']]
        return self.df_sub_unioned
