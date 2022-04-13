from tokenize import group
from kaggle_api import load_data
from dataset import DataSet
from approaches import last_purchased
from partitioned_validation import partitioned_validation, user_grouping_online_and_offline
from recommend_results import RecommendResults
from oneweek_holdout_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


def main():
    # kaggle APIからデータロード
    load_data()
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data()
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
    # とりあえずLast Purchased Item
    df_pred_1 = last_purchased.last_purchased_items(train_transaction=train_df,
                                                    dataset=dataset)
    df_pred_2 = last_purchased.other_colors_of_purchased_item(
        train_transaction=train_df, dataset=dataset)
    df_pred_3 = last_purchased.popular_items_for_each_group(
        train_transaction=train_df, dataset=dataset, grouping_df=group_series)

    print("3")

    # One-week hold-out validationのオフライン評価
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred_1,
                                      grouping=group_series['group'],
                                      approach_name="last_purchased_items"
                                      )
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred_2,
                                      score=score_df,
                                      grouping=group_series['group'],
                                      approach_name="other_colors_of_purchased_item"
                                      )
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred_3,
                                      score=score_df,
                                      grouping=group_series['group'],
                                      approach_name="popular_items_for_each_groupm"
                                      )
    print(score_df.head())


if __name__ == '__main__':
    main()
