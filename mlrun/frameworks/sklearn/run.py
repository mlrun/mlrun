import pandas as pd
import xgboost as xgb
from mlrun.frameworks.sklearn import apply_mlrun
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.datasets import load_boston

def classification():
    classification_models = [
        LogisticRegression(),
        LinearDiscriminantAnalysis(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        GaussianNB()]

    print('Classification')
    # Load Iris Data
    iris = load_iris()

    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = pd.DataFrame(data=iris.target, columns=['species'])

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    for m in classification_models:
        model = apply_mlrun(m, X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)
        model.fit(X_train, y_train.values.reshape(-1, ))
        pred = model.predict(X_test)
        # print(pred)

def regression():
    # Import Regression Models
    regression_models = [linear_model.LinearRegression(),
                         linear_model.Ridge(),
                         linear_model.Lasso(),
                         linear_model.TweedieRegressor()]
    print('Regression')
    # Import
    boston = load_boston()
    bos = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    bos['PRICE'] = boston['target']

    X = bos.drop(['PRICE'], axis=1)
    y = bos['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    for m in regression_models:
        model = apply_mlrun(m, X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # print(pred)


def xgboost_regression():
    print('xgboost_regression')
    # Import
    boston = load_boston()
    bos = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    bos['PRICE'] = boston['target']

    X = bos.drop(['PRICE'], axis=1)
    y = bos['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=10)

    model = apply_mlrun(model, X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    # print(pred)


def xgboost_classification():
    print('xgboost_classification')
    # Import
    iris = load_iris()

    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = pd.DataFrame(data=iris.target, columns=['species'])

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    model = xgb.XGBClassifier()

    model = apply_mlrun(model, X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    # print(pred)

classification()
regression()
xgboost_regression()
xgboost_classification()

