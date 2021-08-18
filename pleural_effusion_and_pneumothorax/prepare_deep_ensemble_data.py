#!~/.venvs/pt/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import os
from scipy.stats import entropy
import numpy as np


def collect_ensemble_results(folder, prefix, dest_dir, arg):
    col = ['ens{}'.format(i) for i in range(arg['n_ensembles'])]
    df_ens = pd.DataFrame(columns=col + ['label'])
    for sid in range(arg['n_ensembles']):
        dir = os.path.join(folder, prefix + '{}.csv.gz'.format(sid))
        try:
            df_ = pd.read_csv(dir, compression='gzip', sep=',')
        except Exception:
            print('Model {} result not found'.format(sid))
            continue
        df_ens['ens{}'.format(sid)] = df_['prob'].to_numpy()
    df_ens['label'] = df_['label'].to_numpy()
    df_ens.to_csv(dest_dir, index=False, compression='gzip')
    print('size:', df_ens.shape)
    return df_ens


def entropy_binary_class(i_preds):
    # Calculate diagnostic entropy per sample
    pk_1 = i_preds.sum() / len(i_preds)
    pk_2 = 1 - pk_1
    pk = np.array([pk_1, pk_2])
    information_entropy = entropy(pk)
    return information_entropy


def add_entropies(df_data):
    d_entropy = []
    e_entropy = []
    for i in range(df_data.shape[0]):
        # Assuming binary output of dignostic neural network are softmax activated
        preds = np.where(df_data.loc[i].to_numpy()[:-1] > 0.5, 1, 0)
        d_entropy.append(entropy_binary_class(preds))
        e_entropy.append(entropy(df_data.loc[i].to_numpy()[:-1]))
    df_data['diagnostic_entropy'] = d_entropy
    df_data['ensemble_entropy'] = e_entropy
    return df_data


def triage_thresholds(df_1, df_2):
    # df_1 for model, df_2 for rejected samples
    if not df_2.empty:
        df_expert = df_2.copy()
        df_expert['pred_y'] = df_expert['true_y']
        df_overall = df_1.append(df_expert)
        deferred_size = df_2.shape[0]
    else:
        df_overall = df_1.copy()
        deferred_size = 0

    deferred_ratio = deferred_size / float(df_overall.shape[0])

    if not df_1.empty:
        model_size = df_1.shape[0]
        model_scores = get_scores(df_1.true_y.tolist(), df_1.pred_y.tolist())
    else:
        model_size = 0
        model_scores = [None, None, None, None]

    overall_scores = get_scores(df_overall.true_y.tolist(), df_overall.pred_y.tolist())

    result = [deferred_ratio, deferred_size, model_size]
    result.extend(overall_scores)
    result.extend(model_scores)

    return result


def get_entropy_triage(test_data, thresholds, entropy_str, save_to_file):
    df_score = pd.DataFrame(columns=['Threshold', 'Deferred_Ratio', 'Deferred_Size', 'Auto_Size', \
                                     'Overall_F1', 'Overall_Accuracy', 'Overall_Sensitivity', 'Overall_Specificity', \
                                     'Model_F1', 'Model_Accuracy', 'Model_Sensitivity', 'Model_Specificity'],
                            dtype=object)
    i = 0
    df = pd.DataFrame(
        {'true_y': test_data['label'].tolist(),
         'pred_y': np.where(test_data['ens0'] > 0.5, 1, 0),
         't_entropy': test_data[entropy_str].tolist()
         })
    for threshold in thresholds:
        df_1 = df[df['t_entropy'] <= threshold]
        df_2 = df[df['t_entropy'] > threshold]
        result = triage_thresholds(df_1, df_2)
        df_score.loc[i] = [threshold] + result
        i += 1
    print(df_score)
    df_score.to_csv(save_to_file, index=False)


arg = dict(n_ensembles=50)
read_from_folder = 'outputs'
ensemble_results_for_train_dir = 'outputs/ensemble_results_for_train.csv.gz'
ensemble_results_for_test_dir = 'outputs/ensemble_results_for_test.csv.gz'
LDU_train_data_dir = 'outputs/LDU_train_data.csv.gz'
LDU_test_data_dir = 'outputs/LDU_test_data.csv.gz'

ensemble_results_for_train = collect_ensemble_results(read_from_folder, 'output_train',
                                                      ensemble_results_for_train_dir, arg)
ensemble_results_for_test = collect_ensemble_results(read_from_folder, 'output_test',
                                                     ensemble_results_for_test_dir, arg)

train_data = add_entropies(ensemble_results_for_train)
train_data.to_csv(LDU_train_data_dir, index=False, compression='gzip')
test_data = add_entropies(ensemble_results_for_test)
test_data.to_csv(LDU_test_data_dir, index=False, compression='gzip')


#---------------------------------------------------------
# Entropy distribution
df_dist = pd.DataFrame(columns=test_data['ensemble_entropy'].describe().index)
df_dist.loc[0] = test_data['ensemble_entropy'].describe().tolist()
df_dist.loc[1] = test_data['diagnostic_entropy'].describe().tolist()
df_dist.to_csv('outputs/distributions_entropies.csv', index=False)

# Direct triage by diagnostic entropy or ensemble entropy:
thresholds = np.arange(round(df_dist['min'][0], 2) - 0.01, round(df_dist['max'][0], 2) + 0.02, 0.01).tolist()
get_entropy_triage(test_data, thresholds, 'ensemble_entropy', 'outputs/triage_by_ensemble_entropy.csv')

if df_dist['max'][1] > 0:
    thresholds = np.arange(0, round(df_dist['max'][1], 2) + 0.02, 0.01).tolist()
    get_entropy_triage(test_data, thresholds, 'diagnostic_entropy', 'outputs/triage_by_diagnostic_entropy.csv')
