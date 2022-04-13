# 「この商品を買った客は、あの商品もよく買ってるぞ！」という考えをベースにしたレコメンド手法

import os
import pandas as pd
import numpy as np
import json

from sympy import Li
from dataset import DataSet
from typing import List, Dict
from tqdm import tqdm


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)


class OftenBuyThatToo:
    # クラス変数の定義
    INPUT_DIR = r"input"
    DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

    def __init__(self, transaction_train: pd.DataFrame) -> None:
        # インスタンス変数(属性の初期化)
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.transaction_train = transaction_train
        self.OBTT_ages_dict: Dict[str, Dict[int, int]] = {}
        pass

    def create_ranking(self, dataset: DataSet, test_bool=False):
        # 動作確認用の時は...
        if test_bool:
            df = self.transaction_train[:10000].reset_index()
        else:
            df = self.transaction_train.reset_index()

        # 年齢層ごとに別のランキングを作りたいので、客がどの年齢層に所属しているかを示すカラムを作っていく。
        listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
        dataset.dfu['age_bins'] = pd.cut(
            dataset.dfu['age'], listBin).astype(str)

        # 年齢層グルーピングのSeriesを取得
        cus_agebins = dataset.dfu['age_bins'].astype(str)
        print(cus_agebins.head())
        print(len(cus_agebins))

        # 年齢層グルーピングのユニーク値のリストを取得
        self.listUniBins = dataset.dfu['age_bins'].unique().tolist()

        # まずは、「各ユーザが買った商品一覧」のDictを作っていく。
        # 結果格納用のdictをInitialize
        ds_dict_c_a: Dict[str, List[str]] = {}

        # 学習期間の、トランザクション1つ1つに対して処理を実行
        for i in tqdm(range(len(df))):
            # トランザクションのユーザid、アイテムidを取得.
            customer_id = int(df.loc[i, "customer_id_short"])
            article_id = int(df.loc[i, "article_id"])

            # もし既にユーザidがdictのkeyに登録されていれば。。。
            if customer_id in ds_dict_c_a:
                # アイテムidをvalueに追加(=valueはListなので、下記のように連結で書ける)
                ds_dict_c_a[customer_id] = ds_dict_c_a[customer_id] + \
                    [article_id]
            else:
                # 未登録のユーザであれば、key:valueとして新規登録.
                ds_dict_c_a[customer_id] = [article_id]

        # 作成したdictをインスタンス変数に保存
        self.c_a_dict = ds_dict_c_a
        # jsonファイルでしゅつりょく
        with open(os.path.join(OftenBuyThatToo.DRIVE_DIR, "dict_c_a_val.json"), mode="w") as f:
            ds_dict_c_a = json.dumps(ds_dict_c_a, cls=MyEncoder)
            f.write(ds_dict_c_a)

        # 次に、「ある商品を買った客一覧」のDictを作っていく。
        # 結果格納用のdictをInitialize。各年齢層グルーピング毎に、「ある商品を買った客一覧」のDictを格納する。
        ds_dict_a_c: Dict[str, Dict[int, List[int]]]
        ds_dict_a_c = {self.listUniBins[i]: {}
                       for i in range(len(self.listUniBins))}

        # 学習期間の、トランザクション1つ1つに対して処理を実行
        for i in tqdm(range(len(df))):
            # トランザクションのユーザid、アイテムidを取得.
            customer_id = df.loc[i, "customer_id_short"]
            article_id = df.loc[i, "article_id"]
            # ユーザのage_binを取得.
            mask = dataset.dfu['customer_id_short'] == int(customer_id)
            age_bin = dataset.dfu[mask]['age_bins'].values[0]

            # 各年齢層毎の「ある商品を買った客一覧」のDictに、対象アイテムをkeyで登録していく。
            # valueは空のリスト。考えてみると、この方法が可読性高いかも。
            ds_dict_a_c[age_bin][int(article_id)] = []

        # 再度、学習期間の、トランザクション1つ1つに対して処理を実行
        for i in tqdm(range(len(df))):
            customer_id = df.loc[i, "customer_id_short"]
            article_id = df.loc[i, "article_id"]
            # ユーザのage_binを取得.
            mask = dataset.dfu['customer_id_short'] == int(customer_id)
            age_bin = dataset.dfu[mask]['age_bins'].values[0]

            # 前のループ処理でInitializeした、対象アイテム：空のリストに、要素(customer_id)を追加していく。
            ds_dict_a_c[age_bin][int(article_id)] += [int(customer_id)]

        # 作成したdictをインスタンス変数に保存
        self.a_c_dict = ds_dict_a_c

        # 買った商品集計リストを作成しやすくするために、article_idを空間圧縮
        df_a_i = {}
        df_i_a = {}
        # 全アイテムに対して、繰り返し処理
        for i in range(len(dataset.dfi)):
            # article_idと、空間圧縮したindexの対応表dictを生成。
            df_a_i[dataset.dfi.loc[i, "article_id"]] = i
            df_i_a[i] = dataset.dfi.loc[i, "article_id"]

        # いよいよ、「ある商品を買った人が他に買っている商品ランキング」を作成していく。
        df_articles = dataset.dfi.copy()
        # 結果格納用のDataFrameをInitialize
        cols = ["article_id", "pred_id", "confidence"]
        table = pd.DataFrame(index=[], columns=cols)

        # 各年齢グルーピング毎に、繰り返し処理していく。
        for uniBin in self.listUniBins:
            # グルーピングがnanなら次のループへ
            if uniBin == 'nan':
                continue

            # 結果格納用のTemporalな変数をInitialize
            article_id = []
            pred_id = []
            confidence = []

            # 各「あるアイテム」と「ある商品を買った客一覧」毎に、繰り返し処理していく
            for articl, customer_list in tqdm(self.a_c_dict[uniBin].items()):

                # 「ある商品を買った人が他に買っている商品」をカウントするListをInitialize
                count = [0]*len(df_articles)

                # 「ある商品を買った客」毎に繰り返し処理
                for customer in customer_list:
                    # 各ユーザが購入したアイテム毎に繰り返し処理
                    for x in self.c_a_dict[customer]:
                        # 「ある商品を買った人が他に買っている商品」Listにカウント
                        count[df_a_i[x]] += 1

                # 「あるアイテム」のidを100個格納したリスト(?)を生成
                art_list = [articl]*100
                # 「あるアイテム」に対して、「ある商品を買った人が他に買っている商品」カウントを降順に並べ替えたリストを生成。
                pred_list = sorted(range(len(df_articles)),
                                   key=lambda k: count[k], reverse=True)[:100]

                # 一緒に買われるアイテム上位100個に対して繰り返し処理：
                for i in range(len(pred_list)):
                    # 結果格納用のListに格納していく。
                    pred_list[i] = df_i_a[pred_list[i]]

                # カウント自体も信頼度を示す指標としてListに保存
                conf_list = sorted(count, reverse=True)[:100]

                # 結果格納用(全アイテム)のリストに、各アイテムの結果を追加。
                article_id.extend(art_list)
                pred_id.extend(pred_list)
                confidence.extend(conf_list)
                del art_list
                del pred_list
                del conf_list

            table = pd.DataFrame(
                list(zip(article_id, pred_id, confidence)), columns=cols)

            # 最後に結果を保存?
            json_path = os.path.join(
                OftenBuyThatToo.DRIVE_DIR, f"items_of_other_costomers_{uniBin}.json")
            with open(json_path, mode="w") as f:
                ranking = json.dumps(table, cls=MyEncoder)
                f.write(ranking)

    def load_ranking(self):
        """本番レコメンド用のjsonデータをロードする関数
        """

        # 「ある商品を買った人が他に買った商品ランキング」を年齢層グループ毎に読み込み
        for uniBin in self.listUniBins:
            # グルーピングがnanなら次のループへ
            if uniBin == 'nan':
                continue
            json_path = os.path.join(
                OftenBuyThatToo.DRIVE_DIR, f"items_of_other_costomers_{uniBin}.json")
            with open(json_path, mode="r") as f:
                # {"年齢層bin": {アイテムid: [アイテムid, ...]}}
                
                self.OBTT_ages_dict[uniBin] = json.load(f)

        print(self.OBTT_ages_dict.keys())

        # 「ある客が買った商品一覧」の辞書を読み込み
        with open(os.path.join(OftenBuyThatToo.DRIVE_DIR, "dict_c_a_val.json"), mode="r") as f:
            self.c_a_dict = json.load(f)

    def create_recommendation(self, dataset: DataSet):
        """レコメンド結果を作成する(予測する)メソッド。
        """
        # レコメンドの枠として、sample_submissionを生成為ておく。
        sub = dataset.df_sub
        # レコメンドアイテムの個数
        N = 12
        M = 100

        def _take_supposedly_popular_products()->List:
            """
            predictionが12個に満たなかった時や、trainデータに無いcustomer_idがあったときのために、
            総購入数が多い商品のランキングを作る。
            """

            # 学習用データ
            df = self.transaction_train
            # 学習期間で、最も売れた上位N個のリストを作る。これでOBTTのレコメンドの穴を埋める。
            general_pred = df["article_id"].astype(str).value_counts(
            ).sort_values(ascending=True)[-N:].index.to_list()

            return general_pred

        self.general_pred = _take_supposedly_popular_products()

        def _get_recommendation_OBTT()->List[str]:
            """それぞれの客が購入した商品から、
            年齢層毎の「ある商品を買った人が他に買った商品ランキング」を用いて、
            次に買いそうな商品を予測していく。
            具体的には以下。
            - 客が買ったそれぞれの商品について、上記ランキングを元に、
            他に買う商品ランキング1位はM点、2位はM-1点、3位はM-2点、4位はM-3点、
            という感じに配点をつけていく。
            """
            
            # レコメンド結果を格納するリスト
            pred_list = []
            print(self.c_a_dict.keys())
            print(sub['customer_id_short'][0:2])
            # レコメンド対象の各ユーザに対して繰り返し処理
            for customer_id in tqdm(sub['customer_id_short']):

                customer_id = str(customer_id)
                # 「ある客が買った商品一覧」のdictにユーザidが含まれていれば...
                # すなわち、2年間で一度でも購入した事があるユーザなら...
                if customer_id in self.c_a_dict:
                    print('True!')
                    # レコメンド候補のarticle_id:スコアを格納するdictをInitialize
                    purchase_dict = {}
                    # 過去に買った商品のリストを取得
                    past_list = self.c_a_dict[customer_id]

                    # ユーザがどの年齢層グループか取得
                    mask = dataset.dfu['customer_id_short'] == int(customer_id)
                    customer_ageBin = dataset.dfu[mask]['age_bins'].values[0]
                    print(customer_ageBin)

                    # 「ある商品を買った人が他に買った商品ランキング」の年齢層グループを決定
                    ds_dict = self.OBTT_ages_dict[str(customer_ageBin)]
                    print(ds_dict.keys())
                    # 各「過去に買った商品」に対して繰り返し処理：
                    for art_id in past_list:
                        # 各「過去に買った商品」に対して「ある商品を買った人が他に買った商品ランキング」を取得
                        print(ds_dict[str(art_id)])
                        rank_list:List[int] = ds_dict[str(art_id)]
                        # ランキング上位M個に対して繰り返し処理：
                        for j in range(M):
                            # j+1位のアイテムのartcle_idが10桁になるように、左側を0埋め
                            item = str(rank_list[j]).zfill(10)
                            # 対象アイテムが、まだレコメンド候補に含まれていなければ...
                            if item not in purchase_dict:
                                # スコアを格納
                                purchase_dict[item] = M-j
                            else:  # 対象アイテムが、既にレコメンド候補に含まれていれば...
                                # スコアを加点。
                                purchase_dict[item] += M-j

                    # 再び、各「過去に買った商品」に対して繰り返し処理：
                    for art_id in past_list:
                        # 対象アイテムが、レコメンド候補に含まれていれば、そのスコアをゼロに
                        # すなわち、一度買ったアイテム自体はレコメンドしない方針。
                        if str(art_id).zfill(10) in purchase_dict:
                            purchase_dict[str(art_id).zfill(10)] = 0

                    # レコメンド候補：スコアのDictをpd.Series()に変換
                    series = pd.Series(purchase_dict)
                    # スコアが0より大きいアイテムのみを抽出
                    series = series[series > 0]
                    # レコメンド候補から、スコアが大きい上位N個を抽出し、そのindex(=article_id)をリストで取得.
                    sub_list = series.nlargest(N).index.tolist()

                # もし、対象ユーザが学習データ期間で、トランザクションがゼロなら...
                else:
                    # 全体の人気ランキングから補完
                    sub_list = self.general_pred
                
                # 各ユーザのレコメンド商品のリストを、全体のレコメンド結果に保存(list=>strに加工して...)
                pred_list.append(' '.join(sub_list))

            # 全ユーザのレコメンド商品のリスト(List[str])を返す。
            return pred_list

        # レコメンド結果をsubに格納
        sub['predicted'] = _get_recommendation_OBTT()

        # csv出力
        file_path = os.path.join(OftenBuyThatToo.DRIVE_DIR, 'submission_OBTT.csv')
        sub.to_csv(file_path, index=None)

        return sub