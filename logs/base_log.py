from pathlib import Path
import logging
import os
import time
from functools import wraps

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


def create_logger(exp_version: str):
    # ログファイルのパス
    log_file = (os.path.join(DRIVE_DIR, f'log/{exp_version}')).resolve()

    # loggerインスタンスの生成
    logger_ = logging.getLogger(name=exp_version)
    logger_.setLevel(logging.DEBUG)

    # formatterインスタンスの生成
    fmr = logging.Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handlerインスタンスの生成
    fh = logging.FileHandler(log_file)
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(fmt=fmr)

    # stream handlerインスタンスの生成
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(hdlr=fh)
    logger_.addHandler(hdlr=ch)


def get_logger(exp_version):
    return logging.getLogger(exp_version)


def stop_watch(VERSION):

    def _stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            # 現在時刻を開始時間に
            start = time.time()

            result = func(*args, **kargs)

            # 秒数をhour, min, secに分割
            elapsed_time = int(time.time() - start)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            # ログ出力
            get_logger(VERSION).info(
                f'[elapsed_time]\t>> {hour: 0>2}:{minits: 0>2}:{sec:0>2}')

        return wrapper

    return _stop_watch
