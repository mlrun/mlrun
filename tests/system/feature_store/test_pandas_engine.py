



# def test_ingest_pands_engine_onehot(rundb_mock):
#     data = get_data()
#
#     # Import MLRun's Feature Store
#     import mlrun.feature_store as fstore
#     from mlrun.feature_store.steps import OneHotEncoder
#
#     # One Hot Encode the newly defined mappings
#     one_hot_encoder_mapping = {
#         "department": list(data["department"].unique()),
#     }
#     # Define the corresponding FeatureSet
#     data_set = fstore.FeatureSet("fs-new",
#                                  entities=[fstore.Entity("id")],
#                                  description="feature set",
#                                  )
#
#     data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
#     df = fstore.ingest(data_set, data, infer_options=fstore.InferOptions.default(), )
#
#
#     data_set_pandas = fstore.FeatureSet("fs-new-pandas",
#                                         entities=[fstore.Entity("id")],
#                                         description="feature set",
#                                         engine='pandas',
#                                         )
#     data_set_pandas.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
#
#
#     df_pandas = fstore.ingest(data_set_pandas, data, infer_options=fstore.InferOptions.default(), )
#
#     assert df.equals(df_pandas)
#
#
# def get_data():
#     names = ['A', 'B', 'C', 'D', 'E']
#     ages = [33, 4, 76, 90, 24]
#     department = ['IT', 'RD', 'RD', 'Marketing', 'IT']
#     timestamp = [time.time(), time.time(), time.time(), time.time(), time.time()]
#     data = pd.DataFrame({'name': names, 'age': ages, 'department': department, 'timestamp': timestamp},
#                         index=[0, 1, 2, 3, 4])
#     data['id'] = data.index
#
#     return data