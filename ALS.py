from distutils.log import error
from tkinter import W
import numpy as np
import numpy


def get_rating_error(x, w, h):
    '''
    行列内の各要素の誤差の値を計算する関数

    parameters
    ------------
    x: 評価行列Xの各要素X_ui
    w:ユーザ行列Wのu列目(列ベクトル)
    h:アイテム行列Hのi列目(列ベクトル)

    Return
    ---------
    X_uiとX_ui_hat(wとhの内積)の誤差.
    '''
    return x - numpy.dot(w, h)


def get_error(X: np.ndarray, W: np.ndarray, H: np.ndarray, beta):
    '''
    ALSによる行列分解における誤差関数の値を計算する関数

    parameters
    -----------
    X:観測された評価行列.
    W:ユーザ行列(潜在変数×ユーザ)
    H:アイテム行列(潜在変数×アイテム)
    beta:L2正則化における罰則項のハイパーパラメータlambda

    Return
    -------
    ALSによる行列分解における目的関数(誤差関数)の値.
    '''
    # 目的関数(誤差関数)の初期値(数式のΣ)
    error = 0.0
    # 2重(各アイテム、各ユーザ)のforループでXの各要素に対して処理を実行する.
    for u in range(len(X)):
        for i in range(len(X[u])):
            # もし要素がゼロなら
            if X[u][i] == 0:
                # 次のループ処理へ
                continue
            # 目的関数(誤差関数)へ、2乗和誤差の足し合わせ
            # pow()関数はべき乗を計算する関数
            error += pow(get_rating_error(x=X[u][i], w=W[:, u], h=H[:, i]), 2)

    # 全ての要素で誤差を計算=>2乗和誤差を算出し終えたら...
    # L2正則化項を追加
    # np.linalg.norm()関数:行列やベクトルのノルム(原点からの距離)を計算する関数
    error += beta/2.0 * (np.linalg.norm(x=W, ord=2) +
                         np.linalg.norm(x=H, ord=2))
    return error


def matrix_factorization(X: np.ndarray, len_of_latest_variable, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    '''
    ALSによる行列分解を実行する関数

    parameters
    -----------
    X:観測された評価行列.
    len_of_latest_variable:潜在変数の数.
    alpha:学習率
    beta:L2正則化における罰則項のハイパーパラメータlambda
    threshold:学習を終了するかどうかの閾値.

    Return
    -----------
    分解されたユーザ行列W(潜在変数×ユーザ)とアイテム行列H(潜在変数×アイテム)
    '''

    # WとHの初期値を設定
    W = np.random.rand(len_of_latest_variable, len(X))
    H = np.random.rand(len_of_latest_variable, len(X[0]))

    # ループ処理でパラメータ更新を繰り返していく
    for step in range(steps):
        # 2重(各アイテム、各ユーザ)のforループでXの各要素に対して処理を実行する.
        for u in range(len(X)):
            for i in range(len(X[u])):
                # もし要素がゼロなら
                if X[u][i] == 0:
                    # 次のループ処理へ
                    continue
                # 更新に必要なeを求める
                err = get_rating_error(x=X[u][i], w=W[:, u], h=H[:, i])
                # 行列WとHの、X_uiに関わる要素の更新
                for k in range(len_of_latest_variable):
                    W[k][u] += alpha * (2 * err * H[k][i])
                    H[k][i] += alpha * (2 * err * W[k][u])

        # 更新後の目的関数(誤差関数)の値を確認
        error = get_error(X, W, H, beta)
        # 十分に評価行列Xを近似できていれば...
        if error < threshold:
            break

    return W, H


def main():
    # サンプルの評価行列を生成
    X = numpy.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ]
    )

    # 行列分解
    W_hat, H_hat = matrix_factorization(X, len_of_latest_variable=2)
    # WとHから、Xの推定値X\hatを生成.
    X_hat = np.dot(a=W_hat.T, b=H_hat)
    print(X_hat)


if __name__ == '__main__':
    main()
