from mlrun.frameworks.sklearn import apply_mlrun
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def classification():
    classification_models = [
        LogisticRegression(),
        LinearDiscriminantAnalysis(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        GaussianNB()]

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
        print(model,pred)


classification()


