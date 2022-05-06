from config import Config
from idna import valid_contextj
import numpy as np
import pandas as pd
from models.rank_learning_lgbm import RankLearningLgbm
from tqdm import tqdm
from datetime import datetime
from my_class.dataset import DataSet
from utils.just_offline_validation import offline_validation
from utils.partitioned_validation import partitioned_validation, user_grouping_online_and_offline
from utils.oneweek_holdout_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation

from models.negative_sampler_class.static_popularity import NegativeSamplerStaticPopularity
import os
from logs.base_log import create_logger, get_logger, stop_watch
# from logs.time_keeper import stop_watch

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

VERSION = f"rankLearning_depth{Config.max_depth}_tc{Config.num_candidate_train}_tp{Config.num_candidate_predict}_full{Config.use_full_sampling}_candidate_is_{Config.predict_candidate_way_name}"


@stop_watch(VERSION)
def run_validation(val_week_id=104):

    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    if Config.use_full_sampling:
        dataset.read_data(c_id_short=True)
        print(len(dataset.df))
        print(len(dataset.df_sub))

    else:
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
        training_days=Config.training_days_one_week_holdout_validation,
        # how="use_same_season_in_past"
    )
    print(len(train_df))

    # 全ユーザをグルーピング
    grouping_df = user_grouping_online_and_offline(dataset=dataset)

    # レコメンド結果を生成
    model = RankLearningLgbm(transaction_train=train_df, dataset=dataset, val_week_id=val_week_id)
    model.preprocessing()
    model.fit()
    # 特徴量重要度の上位50個をログに出力
    get_logger(VERSION).info(f'feature names of highest 50 feature importance are {model.feature_names_highest50_feature_importance}')
    df_sub = model.create_reccomendation()

    # One-week hold-out validationのオフライン評価
    map_k, mp_k = offline_validation(val_df=val_df, pred_df=df_sub)
    # スコアをロギング
    get_logger(VERSION).info(f'va_week_id is {val_week_id}')
    get_logger(VERSION).info(f'map_k of {VERSION} is ...{map_k}')
    get_logger(VERSION).info(f'mean_precision_k of {VERSION} is ...{mp_k}')
    

    # レコメンド結果を保存

    if val_week_id == 105:
        sub_result_dir = os.path.join(DRIVE_DIR, 'submission_csv')
        df_sub.to_csv(os.path.join(sub_result_dir, f'sub_{VERSION}.csv'), index=False)

    else:
        val_result_dir = os.path.join(
            DRIVE_DIR, f'val_results_{val_week_id}_csv')
        df_sub.to_csv(os.path.join(val_result_dir, f'val_{VERSION}.csv'), index=False)

def run_create_sub():
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    if Config.use_full_sampling:
        dataset.read_data()
    else:
        dataset.read_data_sampled()
    # レコメンド結果を生成
    model = RankLearningLgbm(transaction_train=pd.DataFrame(), dataset=dataset, val_week_id=105)
    model.preprocessing()
    model.fit()
    df_sub = model.create_reccomendation()
    sub_result_dir = os.path.join(DRIVE_DIR, 'submission_csv')
    df_sub.to_csv(os.path.join(sub_result_dir, f'sub_{VERSION}.csv'), index=False)

if __name__ == '__main__':
    create_logger(VERSION)
    get_logger(VERSION).info("メッセージ")
    
    # 実行
    if Config.run_for_submittion:
        run_create_sub()
    elif Config.run_for_submittion==False:
        val_week_ids = [104, 103, 102]
        for val_week_id in val_week_ids:
            run_validation(val_week_id=val_week_id)

