class Config:
    # 以下、クラス変数として、設定を記述していく.
    # 基本設定
    num_recommend_item = 12

    # 本番レコメンドか、検証用レコメンドか
    run_for_submittion = False # bool

    # 5%サンプリングを使うか、フルサンプリングを使うか。
    use_full_sampling = False # bool

    # ランク学習用のCandidate
    num_candidate_train = 15 # 訓練用のCandidateの数
    num_candidate_predict = 15 # 予測用のCandidateの数
    # 予測用のCandidateを、オリジナルの手法を使うか、もしくはどの手法から読み込むか。
    predict_candidate_original = True
    predict_candidate_way_name = 'last_purchased_fullTrue_15Candidates' # Noneだったらオリジナル?


    # lightGBMハイパラ
    # 以下が良く調整されるらしい...(深さはあんまり??)
    boosting_type = 'dart' # 多くは'gbdt'. たまに 'dart' or 'goss'
    n_estimators=200 # 最後らへんに増やす。それまではいじらない。
    num_leaves = 63 # かなり多様だが、中央値だと63らしい。
    learning_rate = 0.1 # 最後らへんに減らす。それまではいじらない。
    feature_fraction = 0.8
    bagging_freq = 1
    bagging_fraction = 0.8
    random_state = 0
    # その他調整されてる事が多いパラメータ
    max_depth = 20 # lgbmのtreeの深さ

