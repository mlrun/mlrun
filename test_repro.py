import os
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"

import pandas as pd
import mlrun

# write a small dataframe up to MinIO as a csv, mimicking the tutorial's artifact
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
target_path = "s3://mlrun/test-dataset.csv"
df.to_csv(target_path, storage_options={
    "key": "minioadmin",
    "secret": "minioadmin",
    "endpoint_url": "http://localhost:9000",
})
print("Uploaded.")

# now try to read it back the way MLRun's packager would
item = mlrun.get_dataitem(target_path)
result_df = item.as_df()
print(result_df)