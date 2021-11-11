#!~/.venvs/pt/bin/python
# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import os
import math
from torch.nn.utils import clip_grad_norm_
import logging
import random
from scipy.stats import entropy


class deep_ensemble_class:
    def __init__(self, n_ensembles, device):
        self.n_ensembles = n_ensembles
        self.device = device
        random.seed(47)
        self.list_seeds = random.sample(range(100), self.n_ensembles)
        self.path_ensemble = 'ensemble_data/'

    def calculate_binned_entropy(self, i_preds, n_classes):
        # i_preds: predictions from deep ensembles for ith data point
        pks = []
        for i in range(n_classes):
            pks.append(float(len(i_preds[i_preds == i]) / self.n_ensembles))
        pks = np.array(pks)
        binned_entropy = entropy(pks)
        return binned_entropy

    def add_entropies(self, df_ens, pred_NP, n_classes, filter_zeros):
        logging.info('calculating uncertainty-related features...')
        if n_classes == 2:  # binary classification
            binned_entropy = []
            continuous_entropy = []
            col = ['ens{}cls1'.format(i) for i in range(self.n_ensembles)]
            df_probs = df_ens[col]
            for i in range(df_ens.shape[0]):
                preds = pred_NP[:, i]
                binned_entropy.append(self.calculate_binned_entropy(preds, n_classes))
                continuous_entropy.append(entropy(df_probs.loc[i]))
            if filter_zeros:
                if not all(v == 0 for v in binned_entropy):
                    df_ens['binned_entropy'] = binned_entropy
                else:
                    logging.info('binned entropy are all zero, will not included in LDU')

                if not all(v == 0 for v in continuous_entropy):
                    df_ens['continuous_entropy'] = continuous_entropy
                else:
                    logging.info('continuous entropy for binary class are all zero, will not included in LDU')
            else:
                df_ens['binned_entropy'] = binned_entropy
                df_ens['continuous_entropy'] = continuous_entropy

        else:  # multi-classificaton
            binned_entropy = []
            for i in range(df_ens.shape[0]):
                preds = pred_NP[:, i]
                binned_entropy.append(self.calculate_binned_entropy(preds, n_classes))

            if filter_zeros:
                if not all(v == 0 for v in binned_entropy):
                    df_ens['binned_entropy'] = binned_entropy
                else:
                    logging.info('binned entropy are all zero, will not included in LDU')
            else:
                df_ens['binned_entropy'] = binned_entropy

            for c in range(n_classes):
                continuous_entropy = []
                col = ['ens{}cls{}'.format(i, c) for i in range(self.n_ensembles)]
                df_probs = df_ens[col]
                for i in range(df_ens.shape[0]):
                    continuous_entropy.append(entropy(df_probs.loc[i]))
                if filter_zeros:
                    if not all(v == 0 for v in continuous_entropy):
                        df_ens['continuous_entropy_cls{}'.format(c)] = continuous_entropy
                    else:
                        logging.info('continuous entropy for class {} are all zero, will not included in LDU'.format(c))
                else:
                    df_ens['continuous_entropy_cls{}'.format(c)] = continuous_entropy
        return df_ens

    def ens_to_dataset(self, df_ens):
        x_NP = df_ens.drop(['label'], axis=1).to_numpy()
        labelNP = df_ens.label.to_numpy()
        x = torch.FloatTensor(x_NP.tolist())
        label = torch.LongTensor(labelNP.tolist())
        dataset = torch.utils.data.TensorDataset(x, label)
        return dataset

    def gether_ensembles(self, base_model_class, trainer, validator):
        df_train_ens = pd.DataFrame()
        df_test_ens = pd.DataFrame()
        train_pred = []
        test_pred = []
        compare_train_order = []
        compare_test_order = []

        for i in range(self.n_ensembles):
            # seed: control initialization state of the model
            seed = self.list_seeds[i]
            logging.info('ensemble {}, seed={} :'.format(i, seed))
            torch.cuda.empty_cache()
            torch.manual_seed(seed)
            model = base_model_class().to(self.device)
            model = trainer(model)
            train_label, train_pred_list, train_prob_NP = validator(model, train=True, shuffle=False)
            test_label, test_pred_list, test_prob_NP = validator(model, train=False, shuffle=False)
            if i != 0:
                if compare_train_order != train_label or compare_test_order != test_label:
                    logging.error('validation orders changed, need to turn off shuffle in validator')
                    return None, None, None, None

            compare_train_order = train_label
            compare_test_order = test_label

            train_pred.append(train_pred_list)
            test_pred.append(test_pred_list)
            for j in range(train_prob_NP.shape[1]):
                df_train_ens['ens{}cls{}'.format(i, j)] = train_prob_NP[:, j].tolist()
                df_test_ens['ens{}cls{}'.format(i, j)] = test_prob_NP[:, j].tolist()

        n_classes = train_prob_NP.shape[1]
        df_train_ens['label'] = train_label
        df_test_ens['label'] = test_label
        train_pred = np.array(train_pred)
        test_pred = np.array(test_pred)

        df_train_ens = self.add_entropies(df_train_ens, train_pred, n_classes, filter_zeros=True)
        df_test_ens = self.add_entropies(df_test_ens, test_pred, n_classes, filter_zeros=False)
        df_test_ens = df_test_ens[df_train_ens.columns.tolist()]

        if not os.path.exists(self.path_ensemble):
            os.makedirs(self.path_ensemble)
        df_train_ens.to_csv(os.path.join(self.path_ensemble, 'train_ensembles.csv'), index=False)
        df_test_ens.to_csv(os.path.join(self.path_ensemble, 'test_ensembles.csv'), index=False)
        pd.DataFrame([n_classes]).to_csv(os.path.join(self.path_ensemble, 'n_classes.csv'), index=False)

        return n_classes, df_train_ens.shape[1]-1, self.ens_to_dataset(df_train_ens), self.ens_to_dataset(df_test_ens)

    def load_ensemble(self):
        df_train_ens = pd.read_csv(os.path.join(self.path_ensemble, 'train_ensembles.csv'), sep=',')
        df_test_ens = pd.read_csv(os.path.join(self.path_ensemble, 'test_ensembles.csv'), sep=',')
        n_classes = pd.read_csv(os.path.join(self.path_ensemble, 'n_classes.csv'), sep=',')
        return int(n_classes.loc[0]), df_train_ens.shape[1] - 1, self.ens_to_dataset(df_train_ens), self.ens_to_dataset(df_test_ens)


