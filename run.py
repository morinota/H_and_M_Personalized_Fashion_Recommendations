from kaggle_api import load_data
from dataset import DataSet
from Last_purchased.last_purchased import last_purchased_items
from partitioned_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation, user_grouping_online_and_offline
from recommend_results import RecommendResults

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


def main():
    # kaggle APIからデータロード
    load_data()
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data()

    # One-week hold-out validation
    val_week_id = 104
    val_df = get_valid_oneweek_holdout_validation(
        transaction_df=dataset.df,  # type: ignore
        val_week_id=val_week_id
    )
    train_df = get_train_oneweek_holdout_validation(
        transaction_df=dataset.df,
        val_week_id=104,
        training_days=31,
        how="from_init_date_to_last_date"
    )

    # 全ユーザをグルーピング
    group_df = user_grouping_online_and_offline(dataset=dataset)

    # レコメンド結果を作成し、RecommendResults結果に保存していく。
    recommend_results_valid = RecommendResults
    # とりあえずLast Purchased Item
    dataset, predicted = last_purchased_items(train_transaction=train_df,
                                              dataset=dataset)


if __name__ == '__main__':
    main()