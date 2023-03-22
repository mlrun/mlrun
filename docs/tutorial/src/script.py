import gc

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# [MLRun] Import MLRun:
import mlrun
from mlrun.frameworks.lgbm import apply_mlrun

# [MLRun] Get MLRun's context:
context = mlrun.get_or_create_ctx("apply-mlrun-tutorial")

# [MLRun] Reading train data from context instead of local file:
train_df = context.get_input("train_set", "./train.csv").as_df()
# train_df =  pd.read_csv('./train.csv')

# Drop rows with null values
train_df = train_df.dropna(how="any", axis="rows")


def clean_df(df):
    return df[
        (df.fare_amount > 0)
        & (df.fare_amount <= 500)
        &
        # (df.passenger_count >= 0) & (df.passenger_count <= 8)  &
        (
            (df.pickup_longitude != 0)
            & (df.pickup_latitude != 0)
            & (df.dropoff_longitude != 0)
            & (df.dropoff_latitude != 0)
        )
    ]


train_df = clean_df(train_df)


# To Compute Haversine distance
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    # Define earth radius (km)
    R_earth = 6371
    # Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    # Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    # Compute haversine distance
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R_earth * np.arcsin(np.sqrt(a))


def sphere_dist_bear(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    # Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    # Compute distances along lat, lon dimensions
    dlon = pickup_lon - dropoff_lon

    # Compute bearing distance
    a = np.arctan2(
        np.sin(dlon * np.cos(dropoff_lat)),
        np.cos(pickup_lat) * np.sin(dropoff_lat)
        - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon),
    )
    return a


def radian_conv(degree):
    """
    Return radian.
    """
    return np.radians(degree)


def add_airport_dist(dataset):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    SOL: Statue of Liberty
    NYC: Newyork Central
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty
    nyc_coord = (40.7141667, -74.0063889)

    pickup_lat = dataset["pickup_latitude"]
    dropoff_lat = dataset["dropoff_latitude"]
    pickup_lon = dataset["pickup_longitude"]
    dropoff_lon = dataset["dropoff_longitude"]

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1])
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon)
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon)
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1])
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)
    pickup_sol = sphere_dist(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1])
    dropoff_sol = sphere_dist(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon)
    pickup_nyc = sphere_dist(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1])
    dropoff_nyc = sphere_dist(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon)

    dataset["jfk_dist"] = pickup_jfk + dropoff_jfk
    dataset["ewr_dist"] = pickup_ewr + dropoff_ewr
    dataset["lga_dist"] = pickup_lga + dropoff_lga
    dataset["sol_dist"] = pickup_sol + dropoff_sol
    dataset["nyc_dist"] = pickup_nyc + dropoff_nyc

    return dataset


def add_datetime_info(dataset):
    # Convert to datetime format
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
    )

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday
    dataset["year"] = dataset.pickup_datetime.dt.year

    return dataset


train_df = add_datetime_info(train_df)
train_df = add_airport_dist(train_df)
train_df["distance"] = sphere_dist(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

train_df["bearing"] = sphere_dist_bear(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)
train_df["pickup_latitude"] = radian_conv(train_df["pickup_latitude"])
train_df["pickup_longitude"] = radian_conv(train_df["pickup_longitude"])
train_df["dropoff_latitude"] = radian_conv(train_df["dropoff_latitude"])
train_df["dropoff_longitude"] = radian_conv(train_df["dropoff_longitude"])


train_df.drop(columns=["key", "pickup_datetime"], inplace=True)

y = train_df["fare_amount"]
train_df = train_df.drop(columns=["fare_amount"])


print(train_df.head())

x_train, x_test, y_train, y_test = train_test_split(
    train_df, y, random_state=123, test_size=0.10
)

del train_df
del y
gc.collect()

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": 4,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "subsample": 0.8,
    "bagging_fraction": 1,
    "max_bin": 5000,
    "bagging_freq": 20,
    "colsample_bytree": 0.6,
    "metric": "rmse",
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
    "scale_pos_weight": 1,
    "zero_as_missing": True,
    "seed": 0,
    "num_rounds": 50000,
}

train_set = lgbm.Dataset(
    x_train,
    y_train,
    silent=False,
    categorical_feature=["year", "month", "day", "weekday"],
)
valid_set = lgbm.Dataset(
    x_test,
    y_test,
    silent=False,
    categorical_feature=["year", "month", "day", "weekday"],
)

# [MLRun] Apply MLRun on the LightGBM module:
apply_mlrun(context=context)

model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=10000,
    early_stopping_rounds=500,
    valid_sets=[valid_set],
)
del x_train
del y_train
del x_test
del y_test
gc.collect()

# [MLRun] Reading test data from context instead of local file:
test_df = context.get_input("test_set", "./test.csv").as_df()
# test_df =  pd.read_csv('./test.csv')
print(test_df.head())
test_df = add_datetime_info(test_df)
test_df = add_airport_dist(test_df)
test_df["distance"] = sphere_dist(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

test_df["bearing"] = sphere_dist_bear(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df["pickup_latitude"] = radian_conv(test_df["pickup_latitude"])
test_df["pickup_longitude"] = radian_conv(test_df["pickup_longitude"])
test_df["dropoff_latitude"] = radian_conv(test_df["dropoff_latitude"])
test_df["dropoff_longitude"] = radian_conv(test_df["dropoff_longitude"])


test_key = test_df["key"]
test_df = test_df.drop(columns=["key", "pickup_datetime"])

# Predict from test set
prediction = model.predict(test_df, num_iteration=model.best_iteration)
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})

# [MLRun] Log the submission instead of saving it locally:
context.log_dataset(key="taxi_fare_submission", df=submission, format="csv")
# submission.to_csv('taxi_fare_submission.csv',index=False)
