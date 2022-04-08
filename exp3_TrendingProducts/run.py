from data import load_data
from preprocessing import DataSet
from train import TrainModel
from partitioned_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation

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



    # 対象ユーザをグルーピング
    
    # 



if __name__ == '__main__':
    main()