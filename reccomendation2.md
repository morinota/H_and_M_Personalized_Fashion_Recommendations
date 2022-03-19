<!-- タイトル：kaggle Competitionの為にImplicit ALS base modelの概要を学ぶ１ -->

# はじめに
Kaggle Competition「H&M Personalized Fashion Recommendations」に参加する為に、レコメンデーションについて勉強する事にしました！

少しずつ、レコメンデーションについて自分なりにまとめていきます。

# ALSアルゴリズム（レコメンデーションエンジン）の応用例
- 潜在的特徴(Latest features)の発見
  - 一部のアイテムは様々なカテゴリにまたがっており、グルーピングの整理が困難.
    - =>消費者がどのようにアイテム(映画など)を分類するかをより良く理解できれば、マーケティング戦略に更に力を加える事ができる.
    - =>ALSはコレを助ける事ができる.
  - ユーザ×アイテムの行列を、2つの行列に因数分解する.
    - ![](image_markdown\ALS因数分解.PNG)
    - 分解後の各カラムとレコードのラベルは以下のようになる.
      - ![](image_markdown\ALS因数分解2.PNG)
    - ラベルのない軸には、**潜在的特徴(Latest features)**が入る.
      - 潜在的特徴の数は、これらの行列の**ランク**と呼ばれる.この場合は3。
      - ランクは指定すべきハイパラ.
      - ![](image_markdown\ALS因数分解3.PNG)
      - **潜在的特徴は、元のパターンから作成されたグループ**を表す.(次元圧縮ってこうやってるのか...!)
      - 分解後の行列において、**潜在的特徴でない方の軸は、各アイテム(ユーザ)がこれらのグループにどの程度分類されるか**を表す.
  - アイテム側の行列の見方の例：
    - ![](image_markdown\ALS因数分解2.PNG)
    - 1つのレコードは、ホラー映画の値が高く、ドラマ映画の値は低い。
    - また他のレコードは、その逆。
    - **元々の行列の各映画(アイテム)の情報を少し知っていれば**、**分解後の潜在的特徴がこれら2つのジャンルを反映している**、と判断できる！
    - ＝＞これにより、**ユーザがこれらの映画(アイテム)をどのように分類するかを数学的に確認**できる!
- アイテムのグループ化
- 次元削減
- 画像圧縮
# おわりに

# 参考
以下の記事を参考にさせていただきました！良記事有り難うございます！
- https://www.kaggle.com/julian3833/h-m-implicit-als-model-0-014
- https://blog.uni-3.app/implicit-als
- https://campus.datacamp.com/courses/recommendation-engines-in-pyspark/what-if-you-dont-have-customer-ratings?ex=4
- Pyspark cheet sheet
  - https://datacamp-community-prod.s3.amazonaws.com/65076e3c-9df1-40d5-a0c2-36294d9a3ca9