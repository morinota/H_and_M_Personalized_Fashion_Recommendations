import imp
from typing import List

def iter_to_str(iterable: List[int]) -> str:
    """ article_idの先頭に0を追加し、各article_idを半角スペースで区切られた文字列を返す関数
     (submitのcsvファイル様式にあわせる為に必要)

    Parameters
    ----------
    iterable : List[int]
        イテラブルオブジェクト。ex. 各ユーザへのレコメンド商品のリスト。

    Returns
    -------
    str
       iterable_str：イテラブルオブジェクトの各要素を" "で繋いで文字列型にしたもの。
    """

    # '''
    # article_idの先頭に0を追加し、各article_idを半角スペースで区切られた文字列を返す関数
    # (submitのcsvファイル様式にあわせる為に必要)

    # parameters
    # ===========
    # iterable(ex. List)：イテラブルオブジェクト。ex. 各ユーザへのレコメンド商品のリスト。

    # return
    # ===========
    # iterable_str(str)：イテラブルオブジェクトの各要素を" "で繋いで文字列型にしたもの。
    # '''
    # Listの各要素の先頭に"0"を追加する
    iterable_add_0 = map(lambda x: str(0) + str(x), iterable)
    # リストの要素を半角スペースで繋いで、文字列に。
    iterable_str = " ".join(iterable_add_0)
    return iterable_str
