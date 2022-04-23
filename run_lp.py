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
def run_validation(val_week_id=104):

    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    # dataset.read_data()
    dataset.read_data_sampled()

    print("1")

    # One-week hold-out validation
    val_df = get_valid_oneweek_holdout_validation(
        dataset=dataset,  # type: ignore
        val_week_id=val_week_id
    )
    train_df = get_train_oneweek_holdout_validation(
        dataset=dataset,
        week_column_exist=False,
        val_week_id=val_week_id,
        training_days=31,
        # how="use_same_season_in_past"
    )

    # 全ユーザをグルーピング
    grouping_df = user_grouping_online_and_offline(dataset=dataset)

    # Last Purchased Itemによるレコメンド結果を生成
    df_sub = last_purchased.last_purchased_items(train_transaction=train_df,
                                                  dataset=dataset)
    

    # One-week hold-out validationのオフライン評価
    score_df = partitioned_validation(val_df=val_df,
                                      pred_df=df_sub,
                                      grouping=grouping_df['group'],
                                      approach_name=VERSION
                                      )
    # スコアをロギング
    get_logger(VERSION).info(f'va_week_id is {val_week_id}')
    get_logger(VERSION).info('\t' + score_df.to_string().replace('\n', '\n\t'))

    # レコメンド結果を保存

    if val_week_id == 105:
        sub_result_dir = os.path.join(DRIVE_DIR, 'submission_csv')
        df_sub.to_csv(os.path.join(sub_result_dir, f'sub_{VERSION}.csv'))

    else:
        val_result_dir = os.path.join(
            DRIVE_DIR, f'val_results_{val_week_id}_csv')
        df_sub.to_csv(os.path.join(val_result_dir, f'val_{VERSION}.csv'))


if __name__ == '__main__':
    create_logger(VERSION)
    get_logger(VERSION).info("メッセージ")
    val_week_ids = [104, 103, 102]
    for val_week_id in val_week_ids:
        run_validation(val_week_id=val_week_id)