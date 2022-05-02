import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from my_class.dataset import DataSet
from models.last_purchased_class import LastPurchasedItrems
from utils.partitioned_validation import partitioned_validation, user_grouping_online_and_offline
from utils.oneweek_holdout_validation import get_train_oneweek_holdout_validation, get_valid_oneweek_holdout_validation
from utils.just_offline_validation import offline_validation
import os
from logs.base_log import create_logger, get_logger, stop_watch
from config import Config
from feature import create_user_features, create_item_features
# from logs.time_keeper import stop_watch

DRIVE_DIR = r'/content/drive/MyDrive/Colab Notebooks/kaggle/H_and_M_Personalized_Fashion_Recommendations'


def main():

    # create_user_features.create_user_features()
    create_item_features.create_items_features()


if __name__ == '__main__':
    main()
