import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from my_class.dataset import DataSet
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Union


def _create_is_null_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    pass

def _create_is_zero_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    pass

def _create_is_not_zero_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    pass

def _create_over_0point5_column(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    pass