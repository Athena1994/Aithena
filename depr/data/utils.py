import random
import pandas as pd

from aithena.data.data_provider import ChunkType

from .technical_indicators.collection import IndicatorCollection


class CandleInterval:
    ONE_MINUTE = "ONE_MINUTE"
    TEN_MINUTES = "TEN_MINUTES"
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"


# v: splits a dataframe into chunks based on time differences
def split_time_chunks(df: pd.DataFrame) -> list[tuple[int, int]]:

    # ensure time column is in datetime format
    if df['time'].dtype != 'datetime64[ns]':
        df['time'] = pd.to_datetime(df['time'])

    time_diff = df['time'].diff()
    diff_av = time_diff.min() * 2

    split_indices = time_diff[time_diff > diff_av].index
    split_indices = split_indices.append(pd.Index([len(df)]))

    # make list of tuples with start index and lenght of each chunk
    split_list = []
    start_ix = 0
    for end_ix in split_indices:
        split_list.append((start_ix, end_ix - start_ix))
        start_ix = end_ix

    return split_list


def assign_chunk_ids(df: pd.DataFrame, split_list: list[tuple[int, int]],
                     chunk_size: int, skip_cnt: int,
                     start_id=0) -> pd.DataFrame:
    chunk_id = start_id
    for start_ix, block_len in split_list:
        df.loc[start_ix: start_ix + block_len - 1, 'chunk'] = -1
        for i in range((block_len-skip_cnt) // chunk_size):
            chunk_ix = start_ix + skip_cnt + i*chunk_size
            df.loc[chunk_ix: chunk_ix + chunk_size-1, 'chunk'] = chunk_id
            chunk_id += 1
    df.loc[df['chunk'].isna(), 'chunk'] = -1
    return df


def training_split(chunk_cnt: int,
                   tr_val_ratio: float,
                   test_chunk_cnt: int,
                   current_cnts: tuple = None) -> list[int]:
    # create list with chunk_cnt elements with values 0, 1, 2
    # 0: training, 1: validation, 2: test
    # the values should be randomly distributed with a ratio of tr_val_ratio
    # and a total of test_chunk_cnt test chunks
    val_cnt = chunk_cnt - test_chunk_cnt
    tr_cnt = int((val_cnt * tr_val_ratio)+0.5)
    val_cnt -= tr_cnt

    if current_cnts is not None:
        tr_cnt -= current_cnts[0]
        val_cnt -= current_cnts[1]
        test_chunk_cnt -= current_cnts[2]

    split_list = [ChunkType.TRAINING.as_int()] * tr_cnt
    split_list += [ChunkType.VALIDATION.as_int()] * val_cnt
    split_list += [ChunkType.TEST.as_int()] * test_chunk_cnt
    random.shuffle(split_list)
    return split_list


# finds regions in the data where the indicator needs to be updated
# returns a list of tuples with the start index and the length of the region
def find_indicator_update_regions(data: pd.DataFrame, name: str, skip_cnt: int):
    update_list = []
    split_list = split_time_chunks(data)

    fresh = name not in data.columns  # indicator was not yet added to df

    skip_cnt -= 1  # last sample in skip window is the first to be updated

    for (start_ix, frame_len) in split_list:
        if frame_len > skip_cnt:
            if fresh:
                ix = start_ix
            else:
                val_list = data.loc[start_ix + skip_cnt: start_ix + frame_len,
                                    name]
                val_list = val_list[val_list.isnull()]

                # check if time frame is already complete
                if len(val_list) == 0:
                    continue

                ix = val_list.index.min() - skip_cnt
            update_list.append((ix, start_ix + frame_len - ix))

    return update_list


def apply_indicator(data: pd.DataFrame, required_indicator: dict):
    def convert(x, type):
        if type == 'int':
            return int(x)
        elif type == 'float':
            return float(x)
        elif type == 'str':
            return str(x)
        raise Exception(f"Unknown type {type}")

    if ('name' not in required_indicator) or \
       ('params' not in required_indicator):
        raise Exception('Indicator must have a name and params field')

    indicator = IndicatorCollection.get_from_cfg(required_indicator)

    ind_name = indicator.get_unique_id()

    for region in find_indicator_update_regions(data,
                                                ind_name,
                                                indicator.get_skip_cnt()):
        indicator.apply_to_df(data, ind_name, region[0], region[1])
