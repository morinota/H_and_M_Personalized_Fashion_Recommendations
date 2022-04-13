# 「この商品を買った客は、あの商品もよく買ってるぞ！」という考えをベースにしたレコメンド手法

import os
import pandas as pd
\
import numpy as np
import json
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
        # 年齢層グルーピングのユニーク値のリストを取得
        listUniBins = dataset.dfu['age_bins'].unique().tolist()

        # まずは、「各ユーザが買った商品一覧」のDictを作っていく。
        # 結果格納用のdictをInitialize
        ds_dict_c_a: Dict[str, List[str]] = {}

        print(len(df))
        print(df)
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
        ds_dict_a_c = {listUniBins[i]: {} for i in range(len(listUniBins))}

        # 学習期間の、トランザクション1つ1つに対して処理を実行
        for i in tqdm(range(len(df))):
            # トランザクションのユーザid、アイテムidを取得.
            customer_id = df.loc[i, "customer_id_short"]
            article_id = df.loc[i, "article_id"]
            # ユーザのage_binを取得.
            age_bin = cus_agebins[i]
            # 各年齢層毎の「ある商品を買った客一覧」のDictに、対象アイテムをkeyで登録していく。
            # valueは空のリスト。考えてみると、この方法が可読性高いかも。
            ds_dict_a_c[age_bin][int(article_id)] = []

        # 再度、学習期間の、トランザクション1つ1つに対して処理を実行
        # Initializeした、対象アイテム：空のリストに、要素(customer_id)を追加していく。
        for i in tqdm(range(len(df))):
            customer_id = df.loc[i, "customer_id_short"]
            article_id = df.loc[i, "article_id"]
            age_bin = cus_agebins[i]
            ds_dict_a_c[age_bin][int(article_id)] += [int(customer_id)]

        # 作成したdictをインスタンス変数に保存
        self.a_c_dict = ds_dict_c_a

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
        for uniBin in listUniBins:
            # グルーピングがnanなら次のループへ
            if uniBin == 'nan':
                continue

            # 結果格納用のTemporalな変数をInitialize
            article_id = []
            pred_id = []
            confidence = []

            print(type(self.a_c_dict))
            print(uniBin)

            # 各「あるアイテム」と「ある商品を買った客一覧」毎に、繰り返し処理していく
            for articl, coslist in tqdm(self.a_c_dict[uniBin].items()):

                # 「ある商品を買った人が他に買っている商品」をカウントするListをInitialize
                count = [0]*len(df_articles)

                # 「ある商品を買った客」毎に繰り返し処理
                for costomer in coslist:
                    # 各ユーザが購入したアイテム毎に繰り返し処理
                    for x in ds_dict_c_a[costomer]:
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
            # 結果を保存.
            table.to_pickle(os.path.join(OftenBuyThatToo.DRIVE_DIR,
                            f"items_of_other_costomers_{uniBin}.pkl"))

            # 最後に結果を保存?
            with open(f"items_of_other_costomers_{uniBin}.json", mode="w") as f:
                ranking = json.dumps(table, cls=MyEncoder)
                f.write(ranking)

    def load_ranking(self):
        # 本番レコメンド用のjsonデータをロードする関数

        # 「ある商品を買った人が他に買った商品ランキング」
        with open(os.path.join(OftenBuyThatToo.DRIVE_DIR, "items_of_other_costomers.json"), mode="r") as f:
            self.ds_dict = json.load(f)
        # 「ある客が買った商品一覧」の辞書
        with open(os.path.join(OftenBuyThatToo.DRIVE_DIR, "dict_c_a.json"), mode="r") as f:
            self.c_a_dict = json.load(f)

    def create_recommendation(self):
        pass
