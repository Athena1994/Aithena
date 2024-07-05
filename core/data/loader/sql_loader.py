import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from  .data_loader import OHCLLoader


class SQLOHCLLoader(OHCLLoader):

    def __init__(self, sql_user: str, sql_pw: str,
                 sql_server: str, sql_db: str, sql_table: str):
        super().__init__()

        self.engine = create_engine(f'mysql+pymysql://{sql_user}:{sql_pw}@{sql_server}/{sql_db}')
        self.table = sql_table

    def get(self, pair: str, interval: str,
            earliest: datetime.datetime = None,
            latest: datetime.datetime = None) -> pd.DataFrame:
        with self.engine.connect() as connection:
            query = f'SELECT * FROM {self.table} WHERE time_interval="{interval}" AND currency="{pair}"'
            if earliest is not None:
                query += f' AND time >= "{earliest}"'
            if latest is not None:
                query += f' AND time < "{latest}"'

            df = pd.read_sql(text(query), connection)

            df = df.drop(columns=['currency', 'count', 'time_interval', 'vwap'])

            return df

