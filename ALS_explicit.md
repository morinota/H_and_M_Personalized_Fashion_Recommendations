<!-- タイトル：kaggle Competitionの為にImplicit ALS base modelの概要を学ぶ１ -->

# はじめに
1年前にKaggleに登録しましたが、今回初Competitionとして、「H&M Personalized Fashion Recommendations」に参加してみようと思いました(1ヶ月おくれですが...笑)。
データセットはテーブルデータを基本としている為、画像データやテキストデータに疎い私の様な人にも比較的取っつきやすい気がします。
また、**最終的な成果物(提出物)が"顧客へのレコメンド"**という点がよりビジネス的というか、実務(?)に近いような気がする(私は学生なので偏見かもしれませんが...笑)ので、個人的に楽しみです：）

# レコメンドにおけるexplicitとimplicit
レコメンドエンジンは通販サイトや、最近ではメディアを放送するWebサイト等でもよく見られます。
顧客の嗜好データ(好みのデータ)を元にしたレコメンドエンジンにおいて、活用できるデータは大きく以下の2種に分類できるようです。
- explicit(明示的)データ：rating(ex. 星1~5の評価)など、ユーザ自身が作成した各アイテムの**直接的な**評価データ.
- implicit(暗黙的)データ：クリックやサイト訪問、購入等の、コンバージョンに基づき決められる、**間接的な**評価データ.

# 行列分解(Matrix Factorization, MF)って何ぞや？
行動ベース(implicitベース and explicitベース)の協調フィルタリングでは、疎行列$ユーザ \times アイテム$の行列分解(Matrix Factorization)を考える。
- 評価データ($ユーザ \times アイテム$)の疎行列を$X$とする。
- 行インデックスiと列インデックスjがそれぞれ一人のユーザ、1個のアイテムと対応している.
- 行列の$i, j$成分の値$x_{i, j}$は、ユーザiがアイテムjに対して下した評価(=>explixitデータの例)

ファクターの個数(=グルーピングの数)kを与えた時、評価データXをWとHの積に分解する.
(正確には、Xにできる限り近い(近似した)$\hat{X}=W\cdot H^T$を推定する??)
![](image_markdown/ALS因数分解4.JPG)
$$ X \approx W \cdot H^T = \hat{X} \\
X \in R^{m\times n}, W \in R^{m\times k}, H\in R^{n \times k}$$

ここで、
- Xは行がユーザ、列がアイテムである評価(レーティング)行列。
- Wは行がユーザ、列が潜在変数であるユーザ行列。
- Hは行がアイテム、列が潜在変数であるアイテム行列である。
- mはユニークなユーザの数。nはユニークなアイテムの数である。
- kは、指定された潜在変数の数である。

そして、評価行列の推定値$\hat{X}$の各要素(=**ユーザuによって得られるアイテムuの評価**)は以下の様に、$W_u$(＝行ベクトル)と$H_i^T$(＝列ベクトル)の内積として表せる.

$$ \hat{x}_{u, i} = \mathbf{w}_u^T \cdot \mathbf{h}_i \\
= \sum_{f=0}^{k}{H_{u, f} W_{f, i}}$$

ここで、$\mathbf{w}_u^T, \mathbf{h}_i \in R^{k\times 1}$(長さkのベクトル)である。
## 行列分解における目的関数
行列分解を実施する上での目的は、「**真の評価行列と推定された評価行列の誤差を最小限に抑えること**」である。従って目的関数は...
$$ arg min_{H, W} = ||X-\hat{X}|| \\
=||X-W\cdot H^T|| $$

実際には、絶対値ではなく二乗和誤差に正則化項を加えたものを目的変数とし、**「WとHの各要素」、言い換えれば「各ユーザ、各アイテムに対する$\mathbf{w}_u$及び$\mathbf{h}_i$」を既知の評価値(疎行列Xの非ゼロ要素)から学習するのがMatrix Factrozation**である。

なお目的関数は、行列因数分解アルゴリズムによって多少異なる。(ex. 加える正則化項が異なる)

## 行列分解における最適化手法
目的関数を解析的に解く事は困難なので、勾配降下法等の最適化手法を用いて漸近的に最適解を求める。

最適化手法も、行列因数分解アルゴリズムによって多少異なる。

# ALS（交互最小二乗法、Alternating Least Square）アルゴリズムの理論:
行列因数分解アルゴリズムで、最も有名なものの1つが**ALS（Alternating Least Squares）**である.
(他の手法として、トピックモデルやBPR, FunkSVDなどがある.)
- ALSでは目的関数はL2正則化項を用いる。
  - $$ Loss = \sum_{(u,i) \in I} (x_{ui} - \mathbf{w}_u^T \mathbf{h}_i)^2 + L2\\
  = \sum_{(u,i) \in I} (x_{ui} - \mathbf{w}_u^T \mathbf{h}_i)^2 + \lambda \Biggl(\sum_u n_{w_u}|\mathbf{w}_u|^2 + \sum_i n_{h_i}|\mathbf{h}_i|^2\Biggr) $$
  - ここで、
    - $I ：評価値が格納されているインデックスの集合$
    - $ x_{u, i}：評価行列Xのu行目i列目$
    - $ \lambda：正則化項の強さを指定する値(ハイパラ?)$
    - $ n_{w_u}：Iのu行のランク(ユーザuが評価したアイテムの数)$
    - $ n_{h_i}：Iのi行のランク(アイテムiを評価したユーザの数)$

- ALSでは**2つの損失関数を交互に最小化**する.
  - 最初にユーザ行列を固定し、アイテム行列を使用して最急降下法を実行する。
  - 次にアイテム行列を固定し、ユーザ行列を使用して最急降下法を実行する。これを繰り返す。
- 目的関数を最適化するパラメータを探す為の更新式は以下。
  - $$ E_{u, i} = X_{u, i}-\mathbf{w_i}^T \mathbf{h_i} $$
  - $$ W'_{u, k}= W_{u, k} + 2 \times \alpha \cdot E_{u, i} \cdot H_{k, i}$$
  - $$ H'_{i, k}= H_{u, k} + 2 \times \alpha \cdot E_{u, i} \cdot W_{u, k}$$
  - ここで、$\alpha$は学習率?
- ALSは、トレーニングデータの複数のパーテーションに渡って、最急降下法を並行して実行する。


ユーザの総数を $|U|$、 商品の総数を $|D|$ とすると $P$ と $Q$ はそれぞれ $|U| \times K$ 行列と $|D| \times K$ 行列となります。このとき $K$ はモデルに与えられるパラメータで潜在因子の数を表します。

このとき$\hat{R}$の各要素である、ユーザ $u$ の商品 $i$ に対するレーティングの推定値 $\hat{r}_{ui}$ は

$$ \hat{r}_{ui} = \mathbf{p}_u^\mathrm{T} \mathbf{q}_i $$

で与えられます。

ALS では L2 正則化を行います。このL2(誤差平方和＋L2正則化項)を最小化するようなPとQを求める??


ここで $n_{p_u}$ と $n_{q_i}$ はそれぞれユーザ $u$ と商品 $i$ のレーティングの数を表します。また $I_u$ がユーザ $u$ が評価した商品の集合です。この損失関数を $p_{uk}$ で偏微分してみると：

$$ \begin{aligned}\frac{1}{2} \frac{\partial L}{\partial p_{uk}} &= \sum_{i \in I_u}(\mathbf{p}u^\mathrm{T}\mathbf{q}i - r{ui})q{ik} + \lambda n_{p_u} p_{ku} \ &= \sum_{i \in I_u} q_{ik} \mathbf{q}i^\mathrm{T}\mathbf{p}u - \sum{i \in I_u} q{ik}r_{ui} + \lambda n_{p_u} p_{ku}\end{aligned}$$

$1$ から $K$ までの潜在因子 $k$ を並べてベクトルとしてまとめると：

$$ \begin{aligned}\frac{1}{2} \frac{\partial L}{\partial \mathbf{p}u} &= Q{I_u} Q_{I_u}^\mathrm{T} \mathbf{p}u - Q{I_u} R^\mathrm{T} (u, I_n) + \lambda n_{p_u} \mathbf{p}u \ &= (Q{I_u} Q_{I_u}^\mathrm{T} + \lambda n_p E) \mathbf{p}u - Q{I_u} R^\mathrm{T} (u, I_n) \end{aligned}$$

ここで $Q_{I_u}$ は $Q$ の中からユーザ $u$ が評価した商品 $I_u$ に関する行だけを抜き出したもので、$R^\mathrm{T} (u, I_n)$ は $R$ からユーザ $u$ の行の中から商品 $I_u$ に関するものだけを抜き出したベクトルを表します。

$$ \begin{aligned}\frac{\partial L}{\partial \mathbf{p}u} &= 0 \ \Rightarrow ; \mathbf{p}u &= (Q{I_u} Q{I_u}^\mathrm{T} + \lambda n_p E)^{-1} Q_{I_u} R^\mathrm{T} (u, I_n) \ \Rightarrow ; \mathbf{p}_u &= A_u^{-1}V_u \end{aligned}$$

このようにユーザ因子行列 $P$ は $Q$ と $R$ だけに依存して更新できることが分かります。 $q_{ik}$ で偏微分して整理していけば同様に商品因子行列 $Q$ が $P$ と $R$ だけを使って更新できることが示されます。

ALS ではこの式を用いて $P$ と $Q$ を交互に更新し、損失関数が変化しなくなるまで反復します。問題は $A_u$ の逆行列が存在するのか定かでないことですが、この点についてもとの論文にも特に言及がなかったのでよくわかっていません。
# おわりに

# 参考
以下の記事を参考にさせていただきました！良記事有り難うございます！
- https://www.kaggle.com/julian3833/h-m-implicit-als-model-0-014
- https://blog.uni-3.app/implicit-als
- https://campus.datacamp.com/courses/recommendation-engines-in-pyspark/what-if-you-dont-have-customer-ratings?ex=4
- ZOZO　テックブログ
  - https://techblog.zozo.com/entry/2016/07/01/134825
- https://yuku.takahashi.coffee/blog/2020/01/als-for-matrix-factorization
- https://ichi.pro/rekomenda-shisutemu-no-purototaipingusuteppubaisuteppupa-to-2-kyocho-fyiruta-ringu-niokeru-kogo-saisho-ni-jo-als-gyorets-81873980560545
- 正則化の種類と目的 L1正則化 L2正則化について
  - https://ai-trend.jp/basic-study/neural-network/regularization/