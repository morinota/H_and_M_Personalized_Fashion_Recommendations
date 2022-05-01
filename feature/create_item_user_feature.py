from my_class.dataset import DataSet

class FeatureEngineering_item_and_user:
    def __init__(self, dataset:DataSet, transaction_train) -> None:
        self.dataset = dataset
        self.df_train = transaction_train
        
    def _create_

if __name__ == '__main__':
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    # dataset.read_data(c_id_short=True)
    dataset.read_data_sampled()

    # データをDataFrame型で読み込み
    df_transaction = dataset.df
    df_sub = dataset.df_sub  # 提出用のサンプル
    dfu = dataset.dfu  # 各顧客の情報(メタデータ)
    dfi = dataset.dfi  # 各商品の情報(メタデータ)