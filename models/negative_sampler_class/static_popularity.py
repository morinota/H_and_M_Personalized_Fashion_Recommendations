from datetime import timedelta
from typing import Dict, List, Tuple
from flask import Config
import pandas as pd
from more_itertools import last
from my_class.dataset import DataSet
from utils.useful_func import iter_to_str
import numpy as np
from tqdm import tqdm


class NegativeSamplerStaticPopularity:
    def __init__(self, dataset:DataSet, transaction_train:pd.DataFrame) -> None:
        pass

    def create_negative_sampler(self, unique_customer_ids:List, n_negative:int):