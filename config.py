class Config:
    # クラス変数として、設定を記述していく.
    num_recommend_item = 12
    # 5%サンプリングを使うか、フルサンプリングを使うか。
    use_full_sampling = False

    # ランク学習用のCandidate
    # 訓練用のCandidateの数
    num_candidate_train = 15
    num_candidate_predict = 15

    # lightGBMハイパラ
    # 以下が良く調整されるらしい...(深さはあんまり??)
    boosting_type = 'dart' # 多くは'gbdt'. たまに 'dart' or 'goss'
    n_estimators=300
    num_leaves = 63 # かなり多様だが、中央値だと63らしい。
    learning_rate = 0.1
    feature_fraction = 0.8
    bagging_freq = 1
    bagging_fraction = 0.8
    random_state = 0
    # その他調整されてる事が多いパラメータ
    max_depth = 20 # lgbmのtreeの深さ

