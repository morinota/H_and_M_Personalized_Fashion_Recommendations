import os

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

# kaggle.jsonの情報を環境変数に追加
os.environ['KAGGLE_USERNAME'] = "masatomasamasa"
os.environ['KAGGLE_KEY'] = "5530a94bd76bac1415034d9f14cea01f"

# Kaggle APIを通して提出
from kaggle import KaggleApi 

# predict process

def submit(csv_filepath:str, message:str):
    '''
    Kaggle competitionに結果をSubmitする関数
    '''
    # KaggleApiインスタンスを生成
    api = KaggleApi()
    # 認証を済ませる.
    api.authenticate()

    compe_name = "h-and-m-personalized-fashion-recommendations"
    api.competition_submit(file_name=csv_filepath, message=message, competition=compe_name)


# predict something on test dataset
# submit

filepath = os.path.join(DRIVE_DIR, 'submission_csv/sub_rankLearning_depth20_tc12_tp12_fullTrue_candidate_is_StaticPopularity_byfone_Ensembling.csv')

