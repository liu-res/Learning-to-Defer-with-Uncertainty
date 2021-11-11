#!~/.venvs/pt/bin/python
# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import os
import math
from torch.nn.utils import clip_grad_norm_
import logging
from tabulate import tabulate


class ldu_model(nn.Module):

    def __init__(self, n_classes, input_size):
        super(ldu_model, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.hidden_size = 200
        self.output_size = n_classes + 1
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.l3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.l3(x)
        x = F.softmax(x, dim=1)
        return x


class learning_ldu_class:

    def __init__(self, alpha_list, device='cpu', learning_rate=0.0009, epochs=20):
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha_list = alpha_list
        self.n_classes = None
        self.input_size = None
        self.seed = 47
        self.batch_size = 64
        self.defer_loss = nn.CrossEntropyLoss().to(self.device)
        self.pred_loss = nn.CrossEntropyLoss().to(self.device)

    def loss_function(self, outputs, labels, alpha):
        # Assume expert's predictions always correct and the expert cost is a constant
        batch_size = outputs.size()[0]
        rc = torch.tensor([self.n_classes] * batch_size)
        loss = self.defer_loss(outputs[range(batch_size)], rc) + alpha * self.pred_loss(outputs[range(batch_size)], labels)
        return loss

    def start_training(self, model, train_set, alpha):
        # model: the LDU model
        # alpha: the weight on defer option
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
        model = model.to(self.device)
        # setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        # track training loss
        loss_epoch = np.zeros(self.epochs)
        # Training loop
        for e in range(self.epochs):
            running_loss = 0
            for x_var, y_var in train_loader:
                x_var = Variable(x_var)
                y_var = Variable(y_var)
                optimizer.zero_grad()
                out_mini = model(x_var)
                loss = self.loss_function(out_mini, y_var, alpha)
                # compute the gradient based on the loss
                loss.backward()
                # clip gradient to prevent underflow ( check by: out_mini.min())
                # clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

            logging.info('epoch: {}, loss = {}'.format(e, running_loss/len(train_loader)))
            loss_epoch[e] = running_loss/len(train_loader)

        return model

    def predict(self, model, test_set):
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)
        with torch.set_grad_enabled(False):
            pred = []
            label = []
            for x_var, y_var in test_loader:
                out_mini = model(x_var)
                pred.extend(torch.max(out_mini.cpu().data, 1)[1].numpy().astype(int).tolist())
                label.extend(y_var.cpu().data.numpy().astype(float).tolist())

                del x_var
                del out_mini
            return pred, label

    def get_scores(self, test_y, pred_y):
        # test_y: list of true testing labels
        # pred_y: list of predicted labels
        # return f1, recall, precision
        if len(test_y) < 1:
            return [None, None, None, None]
        else:
            f1 = f1_score(test_y, pred_y, average='micro', zero_division=0)
            recall = recall_score(test_y, pred_y, average='micro', zero_division=0)
            precision = precision_score(test_y, pred_y, average='micro', zero_division=0)
            return [f1, recall, precision]

    def triage_scores(self, pred_y, test_y, n_classes):
        df = pd.DataFrame(
            {'true_y': test_y,
             'pred_y': pred_y
             })
        df_1 = df[df.pred_y != n_classes]
        df_2 = df[df.pred_y == n_classes]

        if not df_2.empty:
            df_expert = df_2.copy()
            df_expert['pred_y'] = df_expert['true_y']
            df_overall = df_1.append(df_expert)
            deferred_size = df_2.shape[0]
        else:
            df_overall = df_1.copy()
            deferred_size = 0

        deferred_ratio = deferred_size / float(df.shape[0])
        logging.info('deferred ratio = {}'.format(deferred_ratio))

        if not df_1.empty:
            size_for_automation = df_1.shape[0]
            model_scores = self.get_scores(df_1.true_y.tolist(), df_1.pred_y.tolist())
            logging.info(
                'Automated F1={}, recall={}, precision={}'.format(model_scores[0], model_scores[1], model_scores[2]))
        else:
            size_for_automation = 0
            model_scores = [None, None, None]

        overall_scores = self.get_scores(df_overall.true_y.tolist(), df_overall.pred_y.tolist())

        result = [deferred_ratio, deferred_size, size_for_automation]
        result.extend(overall_scores)
        result.extend(model_scores)

        return result

    def tune_ldu(self, ensemble_funcion):
        self.n_classes, self.input_size, train_set, test_set = ensemble_funcion()
        logging.info('num classes : {},  LDU input size : {}'.format(self.n_classes, self.input_size))

        df_score = pd.DataFrame(columns=['Alpha', 'Deferred_Ratio', 'Deferred_Size', 'Auto_Size', \
                                         'Overall_F1', 'Overall_Recall', 'Overall_Precision', \
                                         'Auto_F1 (micro)', 'Auto_Recall (micro)', 'Auto_Precision (micro)'],
                                dtype=object)
        i = 0
        for alpha in self.alpha_list:
            logging.info('LDU : alpha={}'.format(alpha))
            torch.cuda.empty_cache()
            torch.manual_seed(self.seed)
            model = ldu_model(n_classes=self.n_classes, input_size=self.input_size)

            model = self.start_training(model, train_set, alpha)
            pred, label = self.predict(model, test_set)
            result = self.triage_scores(pred, label, self.n_classes)
            df_score.loc[i] = [alpha] + result
            i += 1

        print(tabulate(df_score, headers='keys', tablefmt='psql'))

