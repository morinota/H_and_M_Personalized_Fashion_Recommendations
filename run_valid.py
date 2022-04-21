from my_class.dataset import DataSet
from my_class.results_class import Results

from utils.partitioned_validation import partitioned_validation, user_grouping_online_and_offline
from utils.oneweek_holdout_validation import get_valid_oneweek_holdout_validation

from logs.base_log import create_logger, get_logger, stop_watch
# from logs.time_keeper import stop_watch

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

VERSION = "partationed_validation_models"


@stop_watch(VERSION)
def validation_eachmodel(val_results: Results, val_df, grouping_df):
    score_df = 0
    for name in val_results.approach_names_list:

        score_df = partitioned_validation(val_df=val_df,
                                          pred_df=val_results.df_sub,
                                          score=score_df,
                                          grouping=grouping_df['group'],
                                          approach_name=name
                                          )

    return score_df


def main():
    # DataSetオブジェクトの読み込み
    dataset = DataSet()
    # DataFrameとしてデータ読み込み
    dataset.read_data()

    # 検証用データを作成
    val_df = get_valid_oneweek_holdout_validation(
        dataset=dataset,  # type: ignore
        val_week_id=104
    )

    # レコメンド結果の読み込み
    val_results = Results()
    val_results.read_val_data()
    val_results.join_results_all_approaches()

   # 全ユーザをグルーピング
    grouping_df = user_grouping_online_and_offline(dataset=dataset)

    # オフラインスコアを検証
    score_df = validation_eachmodel(val_results, val_df, grouping_df)
    # スコアをロギング
    get_logger(VERSION).info('\t' + score_df.to_string().replace('\n', '\n\t'))


if __name__ == '__main__':
    create_logger(VERSION)
    get_logger(VERSION).info("メッセージ")
    main()
