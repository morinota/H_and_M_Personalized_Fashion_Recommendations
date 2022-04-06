import os
import numpy as np
import pandas as pd
import gc

INPUT_DIR = r"C:\Users\Masat\OneDrive - 国立大学法人東海国立大学機構\input"


def get_submissions():
    sub0 = pd.read_csv(os.path.join(INPUT_DIR, 'submission_byfore_ChrisCombination.csv')).sort_values(
        'customer_id').reset_index(drop=True)
    sub1 = pd.read_csv(os.path.join(INPUT_DIR, 'submission_TrendingProducts_.csv')).sort_values(
        'customer_id').reset_index(drop=True)
    sub2 = pd.read_csv(os.path.join(INPUT_DIR, 'submission_exponentialDecay.csv')).sort_values(
        'customer_id').reset_index(drop=True)
    sub3 = pd.read_csv(os.path.join(INPUT_DIR, 'submission_LSTM_sequential.csv')).sort_values(
        'customer_id').reset_index(drop=True)
    sub4 = pd.read_csv(os.path.join(INPUT_DIR, 'submission_LSTM_itemInforFix.csv')).sort_values(
        'customer_id').reset_index(drop=True)
    sub5 = pd.read_csv(os.path.join(INPUT_DIR, 'submission_timefriend.csv')).sort_values(
        'customer_id').reset_index(drop=True)
    sub6 = pd.read_csv(os.path.join(INPUT_DIR, 'submission_RuleBaseByCustomerAge.csv')).sort_values(
        'customer_id').reset_index(drop=True)

    # How many predictions are in common between models
    print((sub0['prediction'] == sub0['prediction']).mean())
    print((sub0['prediction'] == sub1['prediction']).mean())
    print((sub0['prediction'] == sub2['prediction']).mean())
    print((sub0['prediction'] == sub3['prediction']).mean())
    print((sub0['prediction'] == sub4['prediction']).mean())
    print((sub0['prediction'] == sub5['prediction']).mean())
    print((sub0['prediction'] == sub6['prediction']).mean())

    # return sub0, sub1, sub2, sub3, sub4, sub5, sub6


def main():
    get_submissions()


if __name__ == '__main__':
    main()
