<!-- タイトル：kaggle Competitionの為にImplicit ALS base modelの概要を学ぶ１ -->

# 1. はじめに
Kaggle Competition「H&M Personalized Fashion Recommendations」に参加する為に、レコメンデーションについて勉強する事にしました！

少しずつ、レコメンデーションについて自分なりにまとめていきます。
第一回は、レコメンデーションに使用するデータや、レコメンデーションエンジン手法の大分類まとめます。(今回はコーディングパートはありません！)
  
# 2. レコメンドにおける2種類のデータ(explicitとimplicit)
顧客の嗜好データ(好みのデータ)を元にしたレコメンドエンジンにおいて、活用できるデータは大きく以下の2種に分類できるようです。
- explicit(明示的)データ：
  - ユーザ自身が作成した各アイテムの**直接的な(明示的な)**評価データ.
  - ex. 星1~5の評価, Good or Badボタンなど、
- implicit(暗黙的)データ：
  - ユーザ行動の受動的な追跡に基づいて決められる、**間接的な**評価データ.
  - ex. クリックやサイト訪問、購入等の、**アクションの有無、または頻度**

ちなみに、今回参加する「H&M Personalized Fashion Recommendations」では購買(Transaction)データが提供されているので、"Implicitデータ"を用いたレコメンデーションタスクなんですね！

# 3. 2種類のレコメンデーションエンジン
レコメンドエンジンは通販サイトや、最近ではメディアを放送するWebサイト等でもよく見られます。
レコメンドエンジンの手法は大きく2種類に分けられ、どちらも有益なレコメントを提供する事を目的としていますが、そのアプローチは少し異なります。
1. コンテンツベースのフィルタリングエンジン(Content-Based Filtering)
   - 名前通り"コンテンツ"を理解しようとする.
   - **アイテムの特徴に基づいて**レコメンドする.
     - ex.映画の例
       - ジャンル(コメディ、アクション、ドラマ)
       - アニメ or not
       - 言語(英語、日本語、etc.)
       - 公開された年代(1950's, 1980's)
       - 出演俳優
     - 上記のような特徴を示す"タグ"を作成しておく必要がある!
   - もしAさんがあるドラマ形式の日本映画に5つ星の評価を与えれば、レコメンドエンジンはAさんが"同様の特徴を持ったアイテムが好き"と判断し、似た特徴を持つアイテムをレコメンドする.
2. 協調フィルタリングエンジン(Collaborative Filtering)
   - **ユーザの類似性に基づいて**レコメンドする.
   - 手動で作成したタグは必要ない.
   - 類似性(グルーピング)は、ユーザが提供する評価のパターンから数学的に作成される。
     - (強調フィルタリングの原則の多くは、コンテンツベースの手法に適用できる)
   - 
ちなみに、今回参加する「H&M Personalized Fashion Recommendations」では**購買(Transaction)データ**に加えて、**各アイテムのメタデータ**、**各ユーザのメタデータ**が提供されています。
なので、"コンテンツベース"と"協調フィルタリング"の両方の手法を使う事ができそうですね！

# 4. 終わりに
第一回は、レコメンデーションに使用するデータや、レコメンデーションエンジン手法の大分類に関してまとめました。
第二回は、協調フィルタリングで良く用いられるMatrix Factorization、及びその一手法であるALS(Alternating Least Square)についてまとめ、実装してみようと思います。
最後までお読み頂き、ありがとうございました：）

# 5. 参考
以下の記事を参考にさせていただきました！良記事有り難うございます！
- https://www.kaggle.com/julian3833/h-m-implicit-als-model-0-014
- https://blog.uni-3.app/implicit-als
- https://campus.datacamp.com/courses/recommendation-engines-in-pyspark/what-if-you-dont-have-customer-ratings?ex=4
- Pyspark cheet sheet
  - https://datacamp-community-prod.s3.amazonaws.com/65076e3c-9df1-40d5-a0c2-36294d9a3ca9
- DataCamp
  - https://app.datacamp.com/learn