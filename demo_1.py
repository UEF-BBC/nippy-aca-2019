# Code presented here replicates the results reported in the manuscript for example 1. For the sake for clarity and
# readability, visualization of the results has been omitted.
#
# Jari Torniainen, Department of Applied Physics, University of Eastern Finland
# 2019, MIT License

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import svm, metrics
import scipy.stats
import nippy


def build_svm(x, y, random_state=256):
    """ Utility function for constructing the SVM classifier. Model tuning is done using randomized search with 40
        iterations and each iteration is evaluated using 5-fold CV. Classifier is the nu-variant of the SVM algorithm
        with a polynomial kernel function (degrees 1-3).
    """
    svc = svm.NuSVC()
    params = {}
    params['nu'] =  scipy.stats.uniform(loc=0.01, scale=0.99)
    params['gamma'] = scipy.stats.expon(scale=.4)
    params['degree'] = scipy.stats.randint(1, 4)
    params['kernel'] = ['poly']
    constructor = RandomizedSearchCV(svc, params, cv=5, n_iter=40, verbose=False, n_jobs=-1, random_state=random_state)
    constructor.fit(x, y)
    return constructor


if __name__ == "__main__":
    seed = 6969

    # Dataset can be retrieved from:
    # https://www.kaggle.com/fkosmowski/crop-varietal-identification-with-scio
    df = pd.read_csv("Barley.data.csv")

    # Suffle all the samples before we start because they are ordered by barley cultivars
    df = df.sample(frac=1, random_state=seed)

    # We need to do some data wrangling here to get everything into vectors and matrices
    target = df["Predictor"].get_values()
    nir = df.drop("Predictor", axis=1).get_values()
    wave = np.asarray([int(val) for val in df.drop("Predictor", axis=1).columns])

    # Read pipelines and run data through nippy
    pipelines = nippy.read_configuration("demo_1.ini")
    datasets = nippy.nippy(wave, nir.T, pipelines)

    # Split data into training and testing
    train_idx, test_idx = train_test_split(np.arange(nir.shape[0]), test_size=.30, stratify=target, random_state=seed)

    # Initialize dictionary for storing metrics for each pipeline
    data = {
        "Accuracy (train)": [],
        "Accuracy (test)": [],
        "confusion": [],
        "model": [],
        "pipeline": []
    }

    # Iterate through each pipeline
    for dataset, pipeline in zip(datasets, pipelines):
        x_train = dataset[1][:, train_idx].T
        y_train = target[train_idx]

        x_test = dataset[1][:, test_idx].T
        y_test = target[test_idx]

        # Normalize (i.e. scale) all features before building a model
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # Construct SVM classifier with training data
        model_cv = build_svm(x_train, y_train, random_state=seed)
        accuracy_train = model_cv.best_score_
        model = model_cv.best_estimator_
        model.fit(x_train, y_train)

        # Estimate model performance with the test set
        predicted_test = model.predict(x_test)
        accuracy_test = metrics.accuracy_score(y_test, predicted_test)

        # Collect metrics for this round
        data['model'].append(model)
        data["confusion"].append(metrics.confusion_matrix(target[test_idx], predicted_test))
        data["Accuracy (train)"].append(accuracy_train)
        data["Accuracy (test)"].append(accuracy_test)
        data['pipeline'].append(pipeline)

    # Collect all results to a pandas.DataFrame for easier access
    df = pd.DataFrame(data)

    # Print out the maximum accuracy according to test set performance
    max_idx = df['Accuracy (test)'].idxmax()
    print('Maximum test accuracy: {:0.3f}'.format(df['Accuracy (test)'].max()))
    print(pipelines[max_idx])
