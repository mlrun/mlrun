import os
from mlrun import get_or_create_ctx
import pandas as pd
import v3io_frames as v3f
import dask.dataframe as dd
from dask.distributed import Client

v3c = v3f.Client('framesd:8081', container='bigdata')
dask_client = Client()

def format_df_from_tsdb(shards, df):
    df.index.names = ['timestamp', 'company', 'data_center', 'device']
    df = df.reset_index()
    df = dd.from_pandas(df, npartitions=shards)
    return df


def get_data_tsdb(metrics_table, shards):
    df = v3c.read(backend='tsdb',
                  query=f'select cpu_utilization, latency, packet_loss, throughput, is_error from {metrics_table}',
                  start=f'now-2h', end='now', multi_index=True)
    df = format_df_from_tsdb(shards, df)
    return df


def get_data_parquet(metrics_table, shards):
    mpath = [os.path.join(metrics_table, file) for file in os.listdir(metrics_table)]
    latest = max(mpath, key=os.path.getmtime)
    df = pd.read_parquet(latest)
    df = format_df_from_tsdb(shards, df)
    return df


def create_rolling_featuers(df, window_size: int):
    features = df.copy()
    features['key'] = features.apply(lambda row: f'{row["company"]}_{row["data_center"]}_{row["device"]}', axis=1,
                                     meta=features.compute().dtypes)
    features.set_index('key')
    features["cpu_utilization"] = features.cpu_utilization.rolling(window=window_size).mean()
    features["latency"] = features.latency.rolling(window=window_size).mean()
    features["packet_loss"] = features.packet_loss.rolling(window=window_size).mean()
    features["throughput"] = features.throughput.rolling(window=window_size).mean()
    features["is_error"] = features.is_error.rolling(window=window_size).max()

    features = features.dropna()
    features = features.drop_duplicates()
    return features


def save_to_tsdb(features_table, features: pd.DataFrame):
    v3c.write('tsdb', features_table, features)


def save_to_parquet(features_table, df: pd.DataFrame):
    print('Saving features to Parquet')

    df = df.reset_index()
    df['timestamp'] = df.loc[:, 'timestamp'].astype('datetime64[ms]')
    df = df.set_index(['timestamp', 'company', 'data_center', 'device'])

    first_timestamp = df.index[0][0].strftime('%Y%m%dT%H%M%S')
    last_timestamp = df.index[-1][0].strftime('%Y%m%dT%H%M%S')
    filename = first_timestamp + '-' + last_timestamp + '.parquet'
    filepath = os.path.join(features_table, filename)
    with open(filepath, 'wb+') as f:
        df.to_parquet(f)


def handler(context, event):

    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    mlctx = get_or_create_ctx('mytask', event=event)
    is_tsdb = mlctx.get_param('is_tsdb', False)
    metrics_table = mlctx.get_input('metrics').url
    features_table = mlctx.get_param('features_table', 'features')
    features_table = os.path.join(mlctx.out_path, features_table)
    shards = mlctx.get_param('shards', 1)

    if is_tsdb:
        v3c.create('tsdb', features_table, attrs={'rate': '1/s'}, if_exists=1)
        raw = get_data_tsdb()
    else:
        if not os.path.exists(features_table):
            os.makedirs(features_table)
        raw = get_data_parquet(metrics_table, shards)

    context.logger.info('data prep started!')

    minute = create_rolling_featuers(raw, 3)
    hour = create_rolling_featuers(raw, 10)
    column_names = {'cpu_utilization': 'cpu_utilization_hourly',
                    'latency': 'latency_hourly',
                    'packet_loss': 'packet_loss_hourly',
                    'throughput': 'throughput_hourly'}
    hour = hour.rename(columns=column_names)

    features_rm = raw.merge(minute, on=['timestamp', 'company', 'data_center', 'device'], suffixes=('_raw', '_minute'))
    features_rm.compute()

    features = features_rm.merge(hour, on=['timestamp', 'company', 'data_center', 'device'],
                                 suffixes=('_raw', '_hourly'))
    features = features.compute()

    features = features.reset_index(drop=True)
    feature_cols = [col for col in features.columns if 'key' in col]
    features = features.drop(feature_cols, axis=1)

    features = features.set_index(['timestamp', 'company', 'data_center', 'device'])

    context.logger.info_with("writing features", filepath=features_table)
    if is_tsdb:
        save_to_tsdb(features_table, features)
    else:
        save_to_parquet(features_table, features)
        mlctx.log_artifact('features', target_path=features_table, upload=False)

    return mlctx.to_json()
