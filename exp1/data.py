from kaggle import KaggleApi
import shutil
import os


def load_data():
    '''
    Kaggle competitionのデータセットを読み込む関数
    '''
    # KaggleApiインスタンスを生成
    api = KaggleApi()
    # 認証を済ませる.
    api.authenticate()

    # コンペ一覧
    print(api.competitions_list(group=None, category=None,
          sort_by=None, page=1, search=None))

    # 特定のコンペのデータを取得
    compe_name = "h-and-m-personalized-fashion-recommendations"

    file_list = api.competition_list_files(competition=compe_name)
    print(file_list)

    # csvファイルだけを抽出したい
    file_list_csv = [file.name for file in file_list if '.csv' in file.name]
    print(file_list_csv)

    # 対象データを読み込み(Onedrive上だったらOKか！)
    INPUT_DIR = r"C:\Users\Masat\OneDrive - 国立大学法人東海国立大学機構\input"
    for file in file_list_csv:
        # 各データを読み込み(.zip形式になる)
        api.competition_download_file(competition=compe_name, file_name=file,
                                      path=INPUT_DIR)
        # zipファイルをunpacking
        shutil.unpack_archive(filename=os.path.join(INPUT_DIR, f'{file}.zip'),
                              extract_dir=INPUT_DIR)


def main():
    load_data()


if __name__ == '__main__':
    main()
