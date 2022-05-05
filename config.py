from venv import create
from numpy import negative


class Config:
    # 以下、クラス変数として、設定を記述していく.

    # =========================================================================
    # 基本設定
    num_recommend_item = 12
    #

    # 本番レコメンドか、検証用レコメンドか
    run_for_submittion = False  # bool

    # 5%サンプリングを使うか、フルサンプリングを使うか。
    use_full_sampling = False # bool

    # ==========================================================================
    # ランク学習用のCandidate
    num_candidate_train = 15  # 訓練用のCandidateの数
    num_candidate_predict = 12  # 予測用のCandidateの数
    num_candidate_valid = 7

    # ==================================================================================
    # 予測用のCandidateを、オリジナルの手法(==None)を使うか、もしくはどの手法から読み込むか。
    # predict_candidate_way_name = f'last_purchased_fullTrue_{num_candidate_predict}Candidates' # Noneだったらオリジナル?
    predict_candidate_way_name = None

    # メモリ調整用のNegative Candidate数
    num_negative_candidate = 7

    # =================================================================================
    # 特徴量の話
    # 特徴量生成
    create_user_features = True
    create_item_features = False
    create_not_lag_features = False
    create_lag_features = True

    use_which_user_features = 'original'
    # use_which_user_features = 'my_fullT'
    use_which_item_features = 'original'
    # use_which_item_features = 'my_fullT'

    # 特徴量の種類
    item_basic_feature_names = [
        # 'article_id',
        # 'product_code', # ユニーク値=36284
        # 'prod_name', # ユニーク値=35628
        'product_type_name',  # ユニーク値=125
        'product_group_name',  # ユニーク値=18
        'graphical_appearance_name',  # ユニーク値=30
        'colour_group_name',  # ユニーク値=50
        'perceived_colour_value_name',  # ユニーク値=8
        'perceived_colour_master_name',  # ユニーク値=20
        'department_name', 'index_code',  # ユニーク値=294
        'index_name',  # ユニーク値=10
        'index_group_name',  # ユニーク値=5
        'section_name',  # ユニーク値=56
        'garment_group_name',  # ユニーク値=21
    ]
    item_numerical_feature_names = [
        # 各アイテムの購入価格に関するカラム
        'mean_item_price',  # 各アイテムの購入価格の平均値
        'std_item_price',  # 各アイテムの購入価格の標準偏差
        'max_item_price',  # 各アイテムの購入価格の最大値
        'min_item_price',  # 各アイテムの購入価格の最小値
        'median_item_price',  # 各アイテムの購入価格の中央値
        'sum_item_price',  # 各アイテムの購入価格の合計値(期間内の売上高)
        'count_item_price',  # 各アイテムの購入回数(期間内の売上個数)

        'max_minus_min_item_price',  # 各アイテムの購入価格の最大値と最小値の差
        'max_minus_mean_item_price',  # 各アイテムの購入価格の最大値と平均値の差
        'mean_minus_min_item_price',  # 各アイテムの購入価格の平均値と最小値の差

        # 各アイテムの購入価格に関する小数点以下と整数値のカラム
        # (最大値、中央値、最小値だけでいいかも。＝＞実際にユーザが見る価格はそれらだし。)
        # (4000円か3980円か、みたいな違いを意図してるから、underpointだけでいいかも!)
        'mean_item_price_under_point', 'mean_item_price_over_point',
        'max_item_price_under_point', 'max_item_price_over_point',
        'min_item_price_under_point', 'min_item_price_over_point',
        'median_item_price_under_point', 'median_item_price_over_point',
        'sum_item_price_under_point', 'sum_item_price_over_point',

        # 各アイテムの購入方法に関するカラム(sumは要らないかも...)
        'item_mean_offline_or_online',  # 各アイテムの購入方法(1 or 2)の平均値(1~2)
        'item_median_offline_or_online',  # 各アイテムの購入方法(1 or 2)の中央値(1 or 2)
        'item_sum_offline_or_online'  # 各アイテムの購入方法(1 or 2)の合計値
    ]

    item_categorical_feature_names = []
    item_lag_feature_names = [
        # 実際のカラム名には、_の後にアイテムサブカテゴリstrが続く.
        'lag1_salescount_',
        'lag2_salescount_',

        'rollmean_5week_salescount_',
        'rollmean_10week_salescount_',
        'rollvar_5week_salescount_',
        'rollvar_10week_salescount_',

        'expanding_mean_salescount_',
        'expanding_var_salescount_',
    ]
    item_target_encoding_features = []

    user_numerical_feature_names = [
        'mean_transaction_price',  # 各ユーザの購入価格の平均値
        'std_transaction_price',  # 各ユーザの購入価格の標準偏差
        'max_transaction_price',  # 各ユーザの購入価格の最大値
        'min_transaction_price',  # 各ユーザの購入価格の最小値
        'median_transaction_price',  # 各ユーザの購入価格の中央値
        'sum_transaction_price',  # 各ユーザの購入価格の合計値
        'max_minus_min_transaction_price',  # 各ユーザの購入価格の最大値と最小値の差
        'max_minus_mean_transaction_price',  # 各ユーザの購入価格の最大値と平均値の差
        'mean_minus_min_transaction_price',  # 各ユーザの購入価格の最小値と平均値の差
        'count_transaction_price',  # 各ユーザのアイテム購入回数

        # 各ユーザの購入価格に関する小数点以下と整数値のカラム
        # (アイテム側にはあってもいいけど、ユーザ側には意味ないかも)
        'mean_transaction_price_under_point',
        'mean_transaction_price_over_point',
        'max_transaction_price_under_point', 'max_transaction_price_over_point',
        'min_transaction_price_under_point', 'min_transaction_price_over_point',
        'median_transaction_price_under_point',
        'median_transaction_price_over_point',
        'sum_transaction_price_under_point', 'sum_transaction_price_over_point',

        # 各ユーザの購入方法に関するカラム(sumは要らないかも...)
        'mean_sales_channel_id',  # 各ユーザのアイテム購入におけるonline or offlineの平均値(1~2)
        'median_sales_channel_id',  # 各ユーザのアイテム購入におけるonline or offlineの中央値(1~2)
        'sum_sales_channel_id'  # 各ユーザのアイテム購入におけるonline or offlineの合計値(1~2)
    ]
    item_categorical_feature_names = []
    user_lag_feature_names = []

    # Feature Importance上位50の特徴量(val_week_id=104における)
    feature_names_highest50_feature_importance = [
        'expanding_mean_salescount_article_id',
        'sum_item_price', 'department_name', 'max_minus_mean_item_price', 'sum_item_price_under_point',
        'rollvar_10week_salescount_article_id', 'sum_item_price_over_point', 'count_transaction_price',
        'product_type_name', 'garment_group_name', 'rollvar_10week_salescount_index_group_name',
        'min_item_price_under_point', 'item_sum_offline_or_online', 'section_no', 'std_item_price',
        'graphical_appearance_name', 'expanding_var_salescount_article_id', 'colour_group_name',
        'sum_sales_channel_id', 'min_item_price', 'sum_transaction_price_over_point',
        'expanding_var_salescount_graphical_appearance_name', 'rollmean_5week_salescount_article_id',
        'expanding_var_salescount_perceived_colour_value_name', 'expanding_mean_salescount_perceived_colour_value_name',
        'lag1_salescount_article_id', 'item_mean_offline_or_online', 'min_transaction_price_over_point',
        'expanding_mean_salescount_index_group_name', 'expanding_var_salescount_product_group_name',
        'min_transaction_price', 'lag2_salescount_article_id',
        'mean_item_price_under_point', 'expanding_mean_salescount_product_group_name',
        'mean_item_price', 'sum_transaction_price',
        'mean_transaction_price', 'expanding_var_salescount_department_name', 'count_item_price',
        'lag1_salescount_index_group_name', 'median_transaction_price', 'max_transaction_price', 'rollvar_10week_salescount_product_type_name', 'min_item_price_over_point', 'rollvar_5week_salescount_article_id', 'max_transaction_price_under_point', 'expanding_mean_salescount_garment_group_name', 'expanding_var_salescount_product_type_name', 'rollmean_10week_salescount_article_id', 'mean_item_price_over_point'
    ]

    # ===========================================================================
    # lightGBMハイパラ
    # 以下が良く調整されるらしい...(深さはあんまり??)
    boosting_type = 'gbdt'  # 多くは'gbdt'. たまに 'dart' or 'goss'
    n_estimators = 200  # 最後らへんに増やす。それまではいじらない。
    num_leaves = 63  # かなり多様だが、中央値だと63らしい。
    learning_rate = 0.1  # 最後らへんに減らす。それまではいじらない。
    feature_fraction = 0.8
    bagging_freq = 1
    bagging_fraction = 0.8
    random_state = 0
    # その他調整されてる事が多いパラメータ
    max_depth = 20  # lgbmのtreeの深さ

    # ===================================================---===================
    # validation
    training_days_one_week_holdout_validation = 31
    grouping_column = 'online_and_offline'  # or 'age_bin'
