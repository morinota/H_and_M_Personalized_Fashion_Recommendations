import time
from functools import wraps
from unittest import result
from base_log import get_logger


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
