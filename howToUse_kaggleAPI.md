# はじめに
1年前にKaggleに登録しましたが、今回初Competitionとして、「H&M Personalized Fashion Recommendations」に参加してみようと思いました(1ヶ月おくれですが...笑)。
データセットはテーブルデータを基本としている為、画像データやテキストデータに疎い私の様な人にも比較的取っつきやすい気がします。
また、最終的な成果物が"顧客へのレコメンドという点が、よりビジネス的というか、実務(?)に近いような気がする(私は学生なので偏見かもしれませんが...笑)ので、個人的に楽しみです：）

さて、Competitionを少しでも効率的に行う為に、Kaggle APIを使用したいと考えました！
Kaggle APIの導入方法やコマンドでの操作方法は多くの方々がすでに記事としてまとめてくれていて、情報の取得が容易だと感じました！
一方で、Pythonのスクリプト上でKaggle APIを操作する方法は、私は少し情報の取得に苦労してしまいました...。
そこで本記事では、**Pythonのスクリプト上でKaggle APIを操作する方法**を私なりにまとめました！
(APIクラスが定義されたソースコードを読めば分かる話なのですが、膨大なメソッドがある為、参考記事を発見した後、確認として読むのみに留まりました...笑)

# PythonスクリプトでKaggleApiインスタンスを生成する.
まず、KaggleApiクラスをImportし、インスタンスを生成します。
```python
from kaggle import KaggleApi 
# KaggleApiインスタンスを生成
api = KaggleApi()
# 認証を済ませる.
api.authenticate()
```
このKaggleApiインスタンスのメソッドを操作する事で、Kaggle APIの様々な操作(データ読み込み、提出、リーダーボードの取得、etc.)を実行できるようです。
ソースコードに飛んでみると似たような名前のメソッドがたくさんあり、結構こんがらがります笑
ex. 
- competition_submit()と、competition_submission()
- competition_download_file()と、competition_download_files()

# PythonスクリプトでKaggle competition名のリストを取得する.
competitions_list()メソッドは、KaggleのCompetitionの一覧をリスト形式で返します.
```python
# コンペ一覧
print(api.competitions_list(group=None, category=None, sort_by=None, page=1, search=None))
```
今回は上記の引数をNoneと指定して、Competition名のみのリストを返すようにしています。
(後でソースコードを見たら、これらの引数のデフォルト引数はNoneでした！)

後のデータ取得および提出のプロセスで、Competitionの正式名称(IDみたいな?)が必要になるので、この手順で、自身が参加する予定のCompetitionの正式名称を取得します.
```python
compe_name = "h-and-m-personalized-fashion-recommendations"
```

# PythonスクリプトでKaggle competitionのデータを取得する.
competition_list_files()メソッドは、Competition名を渡して、"提供されるデータセットのファイル名"のリストを取得します。
また、competition_download_file()メソッドは、Competition名とファイル名(+保存場所のパス)を指定して、Compe用のデータファイルを1つずつ読み込む事ができます。

なおH&M Personalized Fashion Recommendationsでは、テーブルデータに加えて各商品の画像データも提供されており、もし画像データを読み込むとデータ量が膨大になる為、今回は.csv形式のファイルのみを読み込んでいます。
```python
import shutil

# 提供されるデータセットのファイル名をリストで取得.
file_list = api.competition_list_files(competition=compe_name)

# csvファイルだけを抽出したい
file_list_csv = [file.name for file in file_list if '.csv' in file.name]
print(file_list_csv)

# 対象データを読み込み
INPUT_DIR = r'input'
for file in file_list_csv:
    # 各データを読み込み(.zip形式になる)
    api.competition_download_file(competition=compe_name, file_name=file, 
    path=INPUT_DIR)
    # zipファイルをunpacking
    shutil.unpack_archive(filename=os.path.join(INPUT_DIR, f'{file}.zip'), 
    extract_dir=INPUT_DIR)
```
また、保存されるファイル形式が.zipの為、shutilモジュールのunpack_archiveを用いて、読み込んだ.zipファイルを解凍しています。(この保存されるファイル形式に関しては参考記事と異なっていたので、APIのバージョンが変わったのかもしれません！)
# PythonスクリプトでKaggle competitionに結果をSubmitする.
competition_submit()メソッドは、提出対象のファイルパスとCompetition名、およびSubmit時のメッセージを引数として渡す事で、CompetitionへのSubmitを実行してくれます。
```python
from kaggle import KaggleApi 

# predict process

def submit(csv_filepath:str, message:str):
    '''
    Kaggle competitionに結果をSubmitする関数
    '''
    # KaggleApiインスタンスを生成
    api = KaggleApi()
    # 認証を済ませる.
    api.authenticate()

    compe_name = "h-and-m-personalized-fashion-recommendations"
    api.competition_submit(file_name=csv_filepath, message=message, competition=compe_name)

def main():
    # predict something on test dataset

    # submit
    filepath = r'input\sample_submission.csv'
    submit(csv_filepath=filepath, message='submission sample')

if __name__ == '__main__':
    main()
```
実行後にKaggleのWebサイトのリーダーボードを確認すると、実際に自分のスコアが提出されている事が確認できました。
# おわりに
本記事では、Kaggle Competitionを少しでも効率的に行う為に、**Pythonのスクリプト上でKaggle APIを操作する方法**を私なりにまとめました。
スクリプトでAPIの操作を記述する事ができれば、手作業のパートを減らせる事に加えて、例えば"スコア提出の判断の自動化"等、コマンドによるAPI操作よりも柔軟な操作が可能になるように感じました！
最後までお読み頂き、ありがとうございました：）
お互い良いKaggle Lifeを！！
ｖ
# 参考
以下の記事を参考にしました。良記事、ありがとうございました！
- https://www.currypurin.com/entry/kaggle-api-submit
- https://atmarkit.itmedia.co.jp/ait/articles/2203/04/news025.html