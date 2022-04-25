class Config:
    # クラス変数として、設定を記述していく.
    num_recommend_item = 12
    # 5%サンプリングを使うか、フルサンプリングを使うか。
    use_full_sampling = False

    # ランク学習用のCandidate
    # 訓練用のCandidateの数
    num_candidate_train = 15
    num_candidate_predict = 30

    # lightGBMハイパラ
    # lgbmのtreeの深さ
    max_depth = 20
    n_estimators=500