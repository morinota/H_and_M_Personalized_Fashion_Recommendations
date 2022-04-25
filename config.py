class Config:
    # クラス変数として、設定を記述していく.
    num_recommend_item = 12
    # 訓練用のCandidateの数
    num_candidate_train = 15
    num_candidate_predict = 20
    # lgbmのtreeの深さ
    max_depth = 20
    n_estimators=500