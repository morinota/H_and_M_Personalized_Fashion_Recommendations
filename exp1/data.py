from kaggle import KaggleApi 

# KaggleApiインスタンスを生成
api = KaggleApi()
# 認証を済ませる.
api.authenticate()

# 後はメソッドで色々...

# コンペ一覧
print(api.competitions_list(group=None, category=None, sort_by=None, page=1, search=None))

# 特定のコンペのデータを取得
compe_name = "h-and-m-personalized-fashion-recommendations"

file_list = api.competition_list_files(competition=compe_name)
print(file_list)

for file
