import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from re import I
from tokenize import group
from kaggle_api import load_data
from dataset import DataSet
from approaches import last_purchased
from partitioned_validation import partitioned_validation, user_grouping_online_and_offline
from recommend_results import RecommendResults
from oneweek_holdout_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation
from recommend_emsemble import recommend_emsemble
import os
from torch import CudaIntStorageBase

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
    grouping_df = user_grouping_online_and_offline(dataset=dataset)
    print(type(grouping_df))
    print("2")

    # レコメンド結果を作成し、RecommendResults結果に保存していく。
    recommend_results_valid = RecommendResults()
    # とりあえずLast Purchased Item
    df_pred_1 = last_purchased.last_purchased_items(train_transaction=train_df,
                                                    dataset=dataset)
    df_pred_2 = last_purchased.other_colors_of_purchased_item(
        train_transaction=train_df, dataset=dataset)
    df_pred_3 = last_purchased.popular_items_for_each_group(
        train_transaction=train_df, dataset=dataset, grouping_df=grouping_df)

    print("3")

    # One-week hold-out validationのオフライン評価
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred_1,
                                      grouping=grouping_df['group'],
                                      approach_name="last_purchased_items"
                                      )
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred_2,
                                      score=score_df,
                                      grouping=grouping_df['group'],
                                      approach_name="other_colors_of_purchased_item"
                                      )
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred_3,
                                      score=score_df,
                                      grouping=grouping_df['group'],
                                      approach_name="popular_items_for_each_group"
                                      )
    predicted_kwargs = {"last_purchased_items": df_pred_1,
                        "other_colors_of_purchased_item": df_pred_2,
                        "popular_items_for_each_groupm": df_pred_3
                        }
    predicted_weights = [100, 10, 1]
    df_pred_0 = recommend_emsemble(predicted_kwargs=predicted_kwargs, weight_args=predicted_weights,
                                   dataset=dataset, val_week_id=val_week_id)

    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred_0,
                                      score=score_df,
                                      grouping=grouping_df['group'],
                                      approach_name="blend"
                                      )
    print(score_df.head())

    val_result_dir = os.path.join(DRIVE_DIR, 'val_results_csv')
    score_df.to_csv(os.path.join(val_result_dir, 'val_last_purchased.csv'))


if __name__ == '__main__':
    main()
