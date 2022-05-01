import pandas as pd
import datetime as dt

class DatetimeFeature:
    def __init__(self) -> None:
        pass

    def add_holiday_dummy(self, df:pd.DataFrame, datetime_column_name:str):
        """平日/休日のダミー変数を追加するカラム

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        datetime_column_name : str
            _description_
        """
        pass




if __name__ == '__main__':

    date_list = [dt.datetime(2018, 10, 1),
                 dt.datetime(2018, 10, 2),
                 dt.datetime(2018, 10, 3),
                 dt.datetime(2018, 10, 4),
                 dt.datetime(2018, 10, 5)]
    sales_list = [64000, 32000, 28000, 78000, 31800]
    sample_df = pd.DataFrame(
        data={'daily_sales': sales_list,
              't_dat': date_list}
    )
    print(sample_df)
