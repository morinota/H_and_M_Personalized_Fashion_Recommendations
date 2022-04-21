
from models.often_by_that_too import OftenBuyThatToo
from utils.kaggle_api import load_data
from my_class.dataset import DataSet

from utils.partitioned_validation import partitioned_validation, user_grouping_online_and_offline

from utils.oneweek_holdout_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


def main():
    # kaggle APIからデータロード
    # load_data()
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data(c_id_short=True)
    print("1")

    # One-week hold-out validation
    val_week_id = 104
    val_df = get_valid_oneweek_holdout_validation(
        dataset=dataset,  # type: ignore
        val_week_id=val_week_id
    )
    train_df = get_train_oneweek_holdout_validation(
        dataset=dataset,
        val_week_id=104,
        training_days=9999,
        how="from_init_date_to_last_date"
    )
    print(len(train_df))

    print("2")

    # 全ユーザをグルーピング
    group_series = user_grouping_online_and_offline(dataset=dataset)
    print(type(group_series))
    print("2")

    # レコメンド結果を作成し、RecommendResults結果に保存していく。
    recommend_results_valid = RecommendResults()
    # OftenBuyThatToo
    model = OftenBuyThatToo(transaction_train=train_df)
    model.create_ranking(dataset=dataset, test_bool=True)
    model.load_ranking()
    # df_pred = model.create_recommendation(dataset=dataset)

    print("3")

    # One-week hold-out validationのオフライン評価
    score = partitioned_validation(val_df=val_df,
                                   pred_df=df_pred,
                                   grouping=group_series,
                                   approach_name="last_purchased_items"
                                   )
    print(score.head())


if __name__ == '__main__':
    main()

