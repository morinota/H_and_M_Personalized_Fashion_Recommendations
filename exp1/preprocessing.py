import os
import pandas as pd
import implicit


def read_data():
    INPUT_DIR = r"input"

    # ファイルパスを用意
    csv_train = os.path.join(INPUT_DIR, 'transactions_train.csv')
    csv_sub = os.path.join(INPUT_DIR, 'sample_submission.csv')
    csv_users = os.path.join(INPUT_DIR, 'customers.csv')
    csv_items = os.path.join(INPUT_DIR, 'articles.csv')

    # データをDataFrame型で読み込み
    df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=[
                     't_dat'])  # 実際の購買記録の情報
    df_sub = pd.read_csv(csv_sub)  # 提出用のサンプル
    dfu = pd.read_csv(csv_users)  # 各顧客の情報(メタデータ)
    dfi = pd.read_csv(csv_items, dtype={'article_id': str})  # 各商品の情報(メタデータ)

    return df, df_sub, dfu, dfi


def preproccessing(df: pd.DataFrame):
    pass


def _extract_byDay(df):
    df = df[df['t_dat'] > '2020-08-21']
    return df


def _add_id_item_and_user(df, dfu, dfi):
    '''
    # ユーザーとアイテムの両方に0から始まる自動インクリメントのidを割り当てる関数
    '''
    ALL_USERS = dfu['customer_id'].unique().tolist()  # ユーザidのユニーク値のリスト
    ALL_ITEMS = dfi['article_id'].unique().tolist()  # アイテムidのユニーク値のリスト

    # key:0から始まるindex, value:ユーザidのdict
    user_ids = dict(list(enumerate(ALL_USERS)))
    # key:0から始まるindex, value:アイテムidのdict
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    # 辞書内包表記で、keyとvalueをいれかえてる...なぜ?? =>mapメソッドを使う為.
    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    # mapメソッドで置換.
    # 引数にdictを指定すると、keyと一致する要素がvalueに置き換えられる.
    # customer_id : ユーザと一意に定まる文字列, user_id：0～ユニーク顧客数のindex
    df['user_id'] = df['customer_id'].map(user_map)
    # article_id : アイテムと一意に定まる文字列, item_id：0～ユニークアイテム数のindex
    df['item_id'] = df['article_id'].map(item_map)

    return df

def _get_rating_matrix(df:pd.DataFrame):
    '''
    トランザクションデータから評価行列を作成する関数
    '''

def main():
    print(implicit.__version__)


if __name__ == '__main__':
    main()
