from numpy import negative


class Config:
    # 以下、クラス変数として、設定を記述していく.

    # =========================================================================
    # 基本設定
    num_recommend_item = 12
    # 

    # 本番レコメンドか、検証用レコメンドか
    run_for_submittion = False # bool

    # 5%サンプリングを使うか、フルサンプリングを使うか。
    use_full_sampling = True # bool

    #==========================================================================
    # ランク学習用のCandidate
    num_candidate_train = 15 # 訓練用のCandidateの数
    num_candidate_predict = 12 # 予測用のCandidateの数
    num_candidate_valid = 7

    # ==================================================================================
    # 予測用のCandidateを、オリジナルの手法(==None)を使うか、もしくはどの手法から読み込むか。
    # predict_candidate_way_name = f'last_purchased_fullTrue_{num_candidate_predict}Candidates' # Noneだったらオリジナル?
    predict_candidate_way_name = None

    # メモリ調整用のNegative Candidate数
    num_negative_candidate = 7

    # =================================================================================
    # 特徴量の話
    # use_which_user_features = 'original' 
    use_which_user_features = 'my_fullT'

    use_which_item_features = 'original' 



    # ===========================================================================
    # lightGBMハイパラ
    # 以下が良く調整されるらしい...(深さはあんまり??)
    boosting_type = 'gbdt' # 多くは'gbdt'. たまに 'dart' or 'goss'
    n_estimators=200 # 最後らへんに増やす。それまではいじらない。
    num_leaves = 63 # かなり多様だが、中央値だと63らしい。
    learning_rate = 0.1 # 最後らへんに減らす。それまではいじらない。
    feature_fraction = 0.8
    bagging_freq = 1
    bagging_fraction = 0.8
    random_state = 0
    # その他調整されてる事が多いパラメータ
    max_depth = 20 # lgbmのtreeの深さ

    #===================================================---===================
    # validation
    training_days_one_week_holdout_validation = 31
    grouping_column = 'online_and_offline' # or 'age_bin'