
from typing import List, Tuple
import pandas as pd
from sympy import arg
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
from collections import defaultdict


def last_purchased_items(train_transaction: pd.DataFrame, dataset: DataSet) -> pd.DataFrame:

    print(train_transaction.head())
    # 各ユーザの、学習期間内における、最終購入日を取得？
    last_purchase = train_transaction.groupby('customer_id_short')["t_dat"].max(
    ).reset_index().rename(columns={'t_dat': 'last_buy_date'})
    # ->カラム["customer_id_short" ,"last_buy_date"]

    # # 各ユーザの最終購入日を学習用のtransactionデータにマージ
    train_transaction = pd.merge(
        train_transaction, last_purchase, on='customer_id_short', how='left')

    # 「各transactionの購入日と、最終購入日との差」を示す"dif_date"カラムを作成
    train_transaction['dif_date'] = (
        train_transaction["last_buy_date"] - train_transaction["t_dat"])
    # pd.Series.dt属性で、日付データの粒度を変換する(年月日＝＞日へ)
    train_transaction['dif_date'] = train_transaction['dif_date'].dt.days

    # 「最終購入日との差が2週間以内のもののみを抽出 & 購入日の昇順に並び変え.
    train_transaction = train_transaction.loc[train_transaction["dif_date"] < 14]\
        .sort_values(['t_dat'], ascending=False)
    # アイテム×ユーザの重複を取り除く(drop_duplicatesは、指定したカラムの全てが重複した行を削除)
    train_transaction = train_transaction.drop_duplicates(
        ['customer_id_short', 'article_id'])

    # 各ユーザ毎に、最終購入日から二週間以内に購入したアイテム達をまとめる。&(リスト=>strへ変換)&(設定されたindexをカラム化)
    last_purchased_items_df = train_transaction.groupby(
        'customer_id_short')['article_id'].apply(iter_to_str).reset_index()
    # ->カラム=[customer_id_short, article_id]、レコードは各ユーザ。article_idは「最終購入日から二週間以内に購入したアイテム達」を繋げたstr.

    # ユーザの不足分(トランザクションデータに含まれていないユーザ)を追加する
    df_sub = dataset.cid.copy()
    df_sub['last_purchased_items'] = pd.merge(last_purchased_items_df, df_sub,
                                                      on='customer_id_short',
                                                      how='right'
                                                      )["article_id"].fillna('')
    df_sub['prediction'] = df_sub['last_purchased_items']
    # 結果はcustomer_idとpredictedをカラムに持つDataFrameにする。
    df_pred = df_sub[['customer_id_short', 'prediction']].copy()

    # 返値は2つにしておく?
    return df_pred


def other_colors_of_purchased_item(train_transaction: pd.DataFrame, dataset: DataSet) -> pd.DataFrame:

    # 設定した期間のtransactionデータに対して、各アイテムの購入回数を集計?
    item_counts_trainTerm: pd.DataFrame = train_transaction.groupby(
        ['article_id'])["t_dat"].count().reset_index()
    # ->カラム["article_id"k, "t_dat"(=各アイテムの購入回数)]
    # カラム名をt_datからcount_termに変更
    item_counts_trainTerm = item_counts_trainTerm.rename(
        columns={'t_dat': 'count_term'})

    # 「設定した期間内における各アイテムの購入回数」とアイテムメタデータをマージ。
    article_df: pd.DataFrame = pd.merge(
        item_counts_trainTerm, dataset.dfi, on='article_id', how='right')

    # count_term(設定した期間内における各アイテムの購入回数)が多い順にソート & count_termが1以上のアイテムのみ抽出。
    popular_item_ranking = article_df.sort_values(
        'count_term', ascending=False).query('count_term > 0')
    # ここまでで学習期間内の、人気アイテムランキング的なものができた。

    # dictを詳しく定義できる関数?(collectionsモジュール)
    map_to_col = defaultdict(list)

    # 各アイテム毎に繰り返し処理：
    for article_id in popular_item_ranking['article_id'].tolist():
        # product_codeカラムは、article_id(9桁)を粗くしたモノ(上6桁)。(＝＞同種のアイテムの違う色とかっぽい！)
        alike_product_code_mask = popular_item_ranking['product_code'] == (
            article_id // 1000)
        # 直近一週間に購入された各商品に対して、類似商品のarticle_idを1つだけ、List型として、Dictに保存
        map_to_col[article_id] = list(filter(
            lambda x: x != article_id, popular_item_ranking[alike_product_code_mask]["article_id"].tolist()))[:1]

    # 各ユーザのLast_purchased_itemを元に、それらの類似アイテムをレコメンドする。
    def _map_to_variation(s):
        '''
        「各ユーザのlast_purchase」の各商品のリストを入力とし、それらの類似商品を返す関数。apply()メソッド用。
        '''
        def f(item): return iter_to_str(map_to_col[int(item)])
        return ' '.join(map(f, s.split()))

    # 上記関数の処理をapplyで適用。other_colorsのレコメンド結果を生成。
    dataset.df_sub['predicted'] = dataset.df_sub['last_purchased_items'].fillna(
        '').apply(_map_to_variation)

    df_pred = dataset.df_sub[['customer_id_short', 'predicted']].copy()

    # 返値は2つにしておく?
    return df_pred


def popular_items_for_each_group(train_transaction: pd.DataFrame, dataset: DataSet, grouping_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    train_transaction : pd.DataFrame
        _description_
    dataset : DataSet
        _description_
    grouping_df : pd.Series
        カラム=[group(=groupingのカテゴリ変数)]、レコードが各ユーザのDataFrame

    Returns
    -------
    Tuple[DataSet, List[List]]
        _description_
    """
    # group_dfをtransactionデータにマージする事で、各transactionにグルーピングを付与する。
    train_transaction = train_transaction.merge(
        grouping_df, on='customer_id_short', how='left')
    # グループ毎に「設定した期間内における各アイテムの購入回数」をカウントする
    ItemCount_eachGroup = train_transaction.groupby(['group', 'article_id'])[
        "t_dat"].count().reset_index()
    # groupbyの後のリセットインデックス大事だわ！
    # -> カラム=[group, article_id, t_dat]、レコードは各アイテム。

    items = defaultdict(str)
    # 各グループ毎に繰り返し処理：
    for group in ItemCount_eachGroup["group"].unique():
        # 各グループで、「設定した期間内における各アイテムの購入回数」の多い上位12個をリストで取得
        # List＝＞strに変換
        # dictのvalueとして保存。keyはグループを示すカテゴリ変数。
        items[group] = iter_to_str(ItemCount_eachGroup.loc[ItemCount_eachGroup["group"] == group].sort_values(
            't_dat', ascending=False)["article_id"].tolist()[:12])

    # 各ユーザに対して、「グループ別の人気アイテム」を取得。レコメンド結果を保存
    dataset.df_sub["predicted"] = grouping_df["group"].map(items)

    #
    df_pred = dataset.df_sub[['customer_id_short', 'predicted']].copy()

    # 返値は2つにしておく?
    return df_pred
