import os
import pandas as pd

def read_data():
    INPUT_DIR = r'input'
    # ファイルパスを用意
    csv_train = os.path.join(INPUT_DIR, 'transactions_train.csv')
    csv_sub = os.path.join(INPUT_DIR, 'sample_submission.csv')
    csv_users = os.path.join(INPUT_DIR, 'customers.csv')
    csv_items = os.path.join(INPUT_DIR, 'articles.csv')

    # データをDataFrame型で読み込み
    df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat']) # 実際の購買記録の情報
    df_sub = pd.read_csv(csv_sub) # 提出用のサンプル
    dfu = pd.read_csv(csv_users) # 各顧客の情報(メタデータ)
    dfi = pd.read_csv(csv_items, dtype={'article_id': str}) # 各商品の情報(メタデータ)

def main():
    pass

if __name__ == '__main__':
    main()
