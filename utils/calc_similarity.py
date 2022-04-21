from turtle import distance
from typing import Dict, List
from sklearn.neighbors import NearestNeighbors
import numpy as np

# アイテムid(str):画像特徴量(ndarray)のdict
images_features_dict: Dict[str, np.ndarray] = {}
item_num = len(images_features_dict.keys())
feature_num = 512

# shape (n_samples, n_features)に変換
data_array = np.zeros(shape=(item_num, feature_num))
for i, key in enumerate(images_features_dict.keys()):
    feature_vec = images_features_dict[key]

    # 追加
    data_array[i, :] = feature_vec

# k-NNオブジェクトのInitialize
nbrs = NearestNeighbors(n_neighbors=100, metric='cosine')
# fit
nbrs.fit(X=data_array)

# 全てのアイテムに対して、画像特徴量の近傍アイテムを取得
simularity_dict = {}
# 各アイテムに対して、コサイン類似度をベースに近傍アイテムを取得
for i, key in enumerate(images_features_dict.keys()):
    feature_vec = images_features_dict[key]
    #
    distances, indices = nbrs.kneighbors(X=[feature_vec])

    # 保存
    simularity_dict[key] = {'distances': distances, 'indices': indices}

