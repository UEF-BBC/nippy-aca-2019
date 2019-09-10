# Code presented here replicates the results reported in the manuscript for example 2. For the sake for clarity and
# readability, visualization of the results has been omitted.
#
# Jari Torniainen, Department of Applied Physics, University of Eastern Finland
# 2019, MIT License

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import cross_val_score, LabelKFold
import nippy
import pymatreader


DATA_COLUMNS = ['preprocessing', 'n_components', 'r2_test', 'r2_train', 'r2_cv', 'rmse_test', 'rmse_train', 'rmse_cv']


def get_train_test(group, targets=['02G', '04B', '04I']):
    """ Returns the indices of samples that belong to group specified by the 'targets'-variable. By default it returns
        indices for a test set similar to the one used in Prakas et al. (2017). """
    mask = np.zeros(group.shape)
    for target in targets:
        mask[group == target] += 1
    test = np.where(mask > 0)[0]
    train = np.where(mask == 0)[0]
    return train, test


def build_pls_model(x, y, max_components=20, cv_fun=10):
    """ Iterates through 1-20 PLS components to find the optimal model. Optimal model selection criteria is
        cross-validated MSE. Function returns the PLSRegression object.
    """
    neg_mse = []
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n, scale=True)
        cv_results = cross_val_score(pls, x, y, cv=cv_fun, scoring='neg_mean_squared_error')
        neg_mse.append(np.mean(cv_results))
    n_best = np.argmax(neg_mse) + 1
    return PLSRegression(n_components=n_best, scale=True)


def evaluate_pipeline(x_train, y_train, x_test, y_test, groups, label='none'):

    plsr = build_pls_model(x_train, y_train, cv_fun=LabelKFold(groups, n_folds=10))
    plsr.fit(x_train, y_train)
    y_pred_train = plsr.predict(x_train).ravel()
    y_pred_test = plsr.predict(x_test).ravel()

    y_pred_test[y_pred_test < 0] = 0

    results = {}
    results['preprocessing'] = label
    results['n_components'] = plsr.n_components
    results['y_true'] = y_test
    results['y_pred'] = y_pred_test
    results['r2_test'] = r2_score(y_test, y_pred_test)
    results['r2_train'] = r2_score(y_train, y_pred_train)
    results['rmse_test'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    results['rmse_train'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    return results


if __name__ == '__main__':
    # Dataset can be retrieved from:
    # https://www.nature.com/articles/s41597-019-0170-y#MOESM1
    data = pymatreader.read_mat('nirs_and_references.mat')['dataset']
    ref = np.array(data['instantaneous_modulus']) / 1e6
    wave = data['wavelength'][0]

    # We need to do some data wrangling here to get everything into vectors and matrices
    group = ['{:02d}{}'.format(int(joint), ai) for joint, ai in zip(data['joint_id'], data['area_of_interest_id'])]
    group = np.array(group)

    nir = np.empty((812, 0))
    for sample in data['raw_spectra']:
        spectrum = np.mean(sample, axis=0).reshape(-1, 1)
        nir = np.concatenate((nir, spectrum), axis=1)


    # Restricting the analysis to same wavelengths as used in Prakash et al. (2017)
    mask = np.bitwise_and(wave >= 700.61, wave <= 1060.41)
    wave = wave[mask]
    nir = nir[mask, :]

    # Assign samples to test/train sets. Using the same split as in Prakash et al. (2017)
    train_idx, test_idx = get_train_test(group)

    # Running the baseline analysis (i.e. no preprocessing)
    x_train = nir[:, train_idx].T
    y_train = ref[train_idx]
    group_train = group[train_idx]

    x_test = nir[:, test_idx].T
    y_test = ref[test_idx]

    results = pd.DataFrame(columns=DATA_COLUMNS)

    baseline_result = evaluate_pipeline(x_train, y_train, x_test, y_test, group_train, label='baseline')
    results = results.append(baseline_result, ignore_index=True)

    # Generate preprocessing pipelines with nippy and evaluate each pipeline separately.
    pipelines = nippy.read_configuration('example_2.ini')
    datasets = nippy.nippy(wave, nir, pipelines)

    for dataset, pipeline in zip(datasets, pipelines):
        wave_, nir_ = dataset
        x_train = nir_[:, train_idx].T
        x_test = nir_[:, test_idx].T

        result = evaluate_pipeline(x_train, y_train, x_test, y_test, group_train, label=pipeline)
        results = results.append({**result}, ignore_index=True)

    # Print out the maximum accuracy according to test set performance
    max_idx = results['r2_test)'].idxmax()
    print('Maximum test accuracy: {:0.3f}'.format(results['r2_test'].max()))
    print(pipelines[max_idx])
