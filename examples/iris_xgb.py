# Example usage of this function with hyper params
#
# from mlrun import new_function, NewRun
#
# parameters = {
#      "eta":       [0.05, 0.10, 0.20, 0.30],
#      "max_depth": [3, 4, 5, 6, 8, 10],
#      "gamma":     [0.0, 0.1, 0.2, 0.3],
#      }
#
# task = NewRun(handler='handler').with_hyper_params(parameters, 'max.accuracy')
# run = new_function(command='iris_xgb.py').run(task)
# print(run.to_df())
#

import xgboost as xgb
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

dtrain = None
dtest = None
Y_test = None

def load_dataset():
    global dtrain, dtest, Y_test
    iris = load_iris()
    y = iris['target']
    X = iris['data']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)


def xgb_train(context, model_name='iris_v1.bst',
            max_depth=6,
            num_class=10,
            eta=0.2,
            gamma=0.1,
            steps=20):
    global dtrain, dtest, Y_test

    if dtrain is None:
        load_dataset()

    param = {"max_depth": max_depth,
             "eta": eta, "nthread": 4,
             "num_class": num_class,
             "gamma": gamma,
             "objective": "multi:softprob"}

    # Train model
    xgb_model = xgb.train(param, dtrain, steps)

    preds = xgb_model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    context.log_result('accuracy', float(accuracy_score(Y_test, best_preds)))

    os.makedirs('models', exist_ok=True)
    model_file = os.path.join('models', model_name)
    xgb_model.save_model(model_file)
    context.log_artifact('model', src_path=model_file, labels={'framework': 'xgboost'})