import os
import pandas as pd
import numpy as np

from partitioned_validation import DRIVE_DIR, INPUT_DIR

class RecommendResults:
    """各レコメンド手法の結果をまとめて管理するRecommendResultsクラス
    """
    # クラス変数の定義
    DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'
    INPUT_DIR = r"submission_csv"

    def __init__(self) -> None:
        # インスタンス変数(属性の初期化)
        self.ALL_ITEMS = []
        self.ALL_USERS = []
        self.results_dict = {}
        
    def read_results(self):

        # submission.csv達が格納されたDirのパス
        results_dir = os.path.join(DRIVE_DIR, INPUT_DIR)
        # 各submission_○○.csvを読みこんで、インスタンス変数results_dictに保存.
        self.results_dict["lstm_itemInforFix"] = pd.read_csv(os.path.join(results_dir, "submission_LSTM_itemInforFix.csv"))
        self.results_dict["lstm_sequential"] = pd.read_csv(os.path.join(results_dir, "submission_LSTM_sequential.csv"))
        self.results_dict["rulebase_byCustomerAge"] = pd.read_csv(os.path.join(results_dir, "submission_RuleBaseByCustomerAge.csv"))
        self.results_dict["trending_products"] = pd.read_csv(os.path.join(results_dir, "submission_TrendingProducts_.csv"))
        self.results_dict["byfone_chris_Combination"] = pd.read_csv(os.path.join(results_dir, "submission_byfone_ChrisCombination.csv"))
        self.results_dict["exponential_Decay"] = pd.read_csv(os.path.join(results_dir, "submission_exponentialDecay.csv"))
        self.results_dict["time_friend"] = pd.read_csv(os.path.join(results_dir, "submission_timefriend.csv"))
        self.results_dict["last_purchased"] = pd.read_csv(os.path.join(results_dir, "submission_timefriend.csv"))