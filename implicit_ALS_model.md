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
# おわりに

# 参考
以下の記事を参考にさせていただきました！良記事有り難うございます！
- https://www.kaggle.com/julian3833/h-m-implicit-als-model-0-014
- https://blog.uni-3.app/implicit-als
- https://campus.datacamp.com/courses/recommendation-engines-in-pyspark/what-if-you-dont-have-customer-ratings?ex=4