from my_class.dataset import DataSet
from my_class.results_class import Results
import pandas as pd
from utils.partitioned_validation import partitioned_validation, user_grouping_online_and_offline, user_grouping_age_bin
from utils.oneweek_holdout_validation import get_valid_oneweek_holdout_validation
import os
from logs.base_log import create_logger, get_logger, stop_watch
# from logs.time_keeper import stop_watch

val_week_id = 104
DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'
# VERSION = "partationed_validation_models_onlineOrOffline"
VERSION = f"partationed_validation_models_{val_week_id}_ageBin"


@stop_watch(VERSION)
def validation_eachmodel(val_results: Results, val_df, grouping_df) -> pd.DataFrame:
    # 初期値をInitialize
    score_df = 0
    for name in val_results.approach_names_list:
        # 特定のモデルのレコメンド結果だけ抽出
        pred_df = val_results.df_sub[[
            'customer_id', 'customer_id_short', f'{name}']]
        pred_df.rename(columns={f'{name}': 'prediction'}, inplace=True)
        # オフラインスコアの検証
        score_df = partitioned_validation(val_df=val_df,
                                          pred_df=pred_df,
                                          score=score_df,
                                          grouping=grouping_df['group'],
                                          approach_name=name
                                          )
        del pred_df
        print(score_df.columns)

    print(type(score_df))
    # スコアをロギング
    get_logger(VERSION).info('\t' + score_df.to_string().replace('\n', '\n\t'))
    # csvでも保存しておく
    score_df.to_csv(os.path.join(DRIVE_DIR, f'logs/{VERSION}.csv'), index=True)
    return score_df


@stop_watch(VERSION)
def main(val_week_id):
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data()

    # 検証用データを作成
    val_df = get_valid_oneweek_holdout_validation(
        dataset=dataset,  # type: ignore
        val_week_id=val_week_id
    )

    # レコメンド結果の読み込み
    val_results = Results()
    val_results.read_val_data(val_week_id=val_week_id)
    val_results.join_results_all_approaches()

   # 全ユーザをグルーピング
    # grouping_df = user_grouping_online_and_offline(dataset=dataset)
    grouping_df = user_grouping_age_bin(dataset=dataset)


    # オフラインスコアを検証
    score_df = validation_eachmodel(val_results, val_df, grouping_df)
    print(type(score_df))


if __name__ == '__main__':
    create_logger(VERSION)
    get_logger(VERSION).info("メッセージ")
    main(val_week_id)
