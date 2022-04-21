from pathlib import Path
import logging
import os

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'

def create_logger(exp_version:str):
    # ログファイルのパス
    log_file = os.path.join(DRIVE_DIR, f'log/{exp_version}')

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