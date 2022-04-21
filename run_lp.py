import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from my_class.dataset import DataSet
from models import last_purchased
from utils.partitioned_validation import partitioned_validation, user_grouping_online_and_offline
from utils.oneweek_holdout_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation
from utils.recommend_emsemble import recommend_emsemble
import os
from logs.base_log import create_logger, get_logger, stop_watch
# from logs.time_keeper import stop_watch

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

VERSION = "last_purchased"


@stop_watch(VERSION)
def main(val_week_id=104):

    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data()
    print("1")

    # One-week hold-out validation
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

    # 全ユーザをグルーピング
    grouping_df = user_grouping_online_and_offline(dataset=dataset)

    # Last Purchased Itemによるレコメンド結果を生成
    df_pred = last_purchased.last_purchased_items(train_transaction=train_df,
                                                  dataset=dataset)
    '''df_pred_2 = last_purchased.other_colors_of_purchased_item(
        train_transaction=train_df, dataset=dataset)
    df_pred_3 = last_purchased.popular_items_for_each_group(
        train_transaction=train_df, dataset=dataset, grouping_df=grouping_df)
'''

    # One-week hold-out validationのオフライン評価
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_pred,
                                      grouping=grouping_df['group'],
                                      approach_name="last_purchased_items"
                                      )
    # スコアをロギング
    get_logger(VERSION).info('\t' + score_df.to_string().replace('\n', '\n\t'))

'''    score_df = partitioned_validation(val_df=val_df,
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

    # 3種のレコメンドを結合
    predicted_weights = [100, 10, 1]
    df_sub = recommend_emsemble(predicted_kwargs=predicted_kwargs, weight_args=predicted_weights,
                                dataset=dataset, val_week_id=val_week_id)

    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_sub,
                                      score=score_df,
                                      grouping=grouping_df['group'],
                                      approach_name="blend"
                                      )
    # レコメンド結果を保存
    if val_week_id == 104:
        val_result_dir = os.path.join(DRIVE_DIR, 'val_results_csv')
        df_sub.to_csv(os.path.join(val_result_dir, 'val_last_purchased.csv'))
        '''
if val_week_id == 105:
    sub_result_dir = os.path.join(DRIVE_DIR, 'submission_csv')
    df_sub.to_csv(os.path.join(val_result_dir, 'val_last_purchased.csv'))
if __name__ == '__main__':
    create_logger(VERSION)
    get_logger(VERSION).info("メッセージ")
    main(val_week_id=104)
