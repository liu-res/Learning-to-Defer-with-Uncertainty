#!~/.venvs/pt/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import os
import math
from torch.nn.utils import clip_grad_norm_


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Network_ldu(nn.Module):

    def __init__(self):
        super(Network_ldu, self).__init__()
        self.l1 = nn.Linear(arg['input_size'], arg['hidden_size'])
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(arg['hidden_size'], arg['output_size']+1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


def loss_function(outputs, labels, n_classes, alpha):
    # Assume expert's predictions always correct and the expert cost is a constant
    # labels: ground truth, n_classes: number of classes
    batch_size = outputs.size()[0]
    # defer class:rc
    rc = [n_classes] * batch_size
    rc = torch.tensor(rc).to(arg['device'])
    loss = loss_expert(outputs[range(batch_size)], rc) + alpha*loss_model(outputs[range(batch_size)], labels)
    return loss


def test_predict(net, arg, x_val):
    with torch.set_grad_enabled(False):
        # predictions on validation set:
        val_preds = np.empty(0).astype(int)
        for i in range(0, x_val.shape[0], arg['batch_size']):
            x_mini = x_val[i:i + arg['batch_size']].to(arg['device'])
            out_mini = net(x_mini)
            val_preds = np.concatenate([val_preds, torch.max(out_mini.cpu().data, 1)[1].numpy().astype(int)])

            del x_mini
            del out_mini
        return val_preds


def get_scores(true_y: list, pred_y: list):
    # F1, and from Confusion matrix compute Accuracy, sensitivity(TP/P) and specificity(TN/N)
    if len(true_y) == 1:
        f1 = None
    else:
        f1 = f1_score(true_y, pred_y, zero_division=0)

    if (sum(true_y) == len(true_y)) & (true_y == pred_y):
        print('all are positive, and predicted correctly')
        return [f1, 1.0, 1.0, None]
    elif (sum(true_y) == 0) & (true_y == pred_y):
        print('all are negative, and predicted correctly')
        return [f1, 1.0, None, 1.0]
    else:
        cm = confusion_matrix(true_y, pred_y)
        total = sum(sum(cm))
        accuracy = (cm[0, 0] + cm[1, 1]) / float(total)
        sensitivity = cm[0, 0] / float(cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / float(cm[1, 0] + cm[1, 1])
        return [f1, accuracy, sensitivity, specificity]


def triage_scores(predicted_label, y_test, arg):
    df = pd.DataFrame(
        {'true_y': y_test.tolist(),
         'pred_y': predicted_label
         })
    df_1 = df[df.pred_y != arg['n_classes']]
    df_2 = df[df.pred_y == arg['n_classes']]

    if not df_2.empty:
        df_expert = df_2.copy()
        df_expert['pred_y'] = df_expert['true_y']
        df_overall = df_1.append(df_expert)
        deferred_size = df_2.shape[0]
    else:
        df_overall = df_1.copy()
        deferred_size = 0

    deferred_ratio = deferred_size / float(df.shape[0])

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


def train(x_new, y_new, x_test, y_test, arg, alpha):
    # init Network_ldu
    torch.manual_seed(arg['seed'])
    net = Network_ldu().to(arg['device'])
    # setup optimizer
    optimizer = optim.SGD(net.parameters(), lr=arg['learning_rate'], momentum=0.9)
    # track training loss
    loss_epoch = np.zeros(arg['epochs'])

    # Training loop
    for e in range(arg['epochs']):
        batch_ndx = -1
        for i in range(0, x_new.shape[0], arg['batch_size']):
            batch_ndx += 1
            x_mini = x_new[i:i + arg['batch_size']].to(arg['device'])
            y_mini = y_new[i:i + arg['batch_size']].to(arg['device'])
            # Wrap mini batch input_mnist tensor to allow automatic gradient computation when backpropagating
            x_var = Variable(x_mini)
            y_var = Variable(y_mini)
            # resets the gradient of the optimizer
            optimizer.zero_grad()
            # forward propagation:
            net_out = net(x_var)
            # computes the loss
            loss = loss_function(net_out, y_var, arg['n_classes'], alpha)
            # compute the gradient based on the loss
            loss.backward()
            # clip gradient to prevent underflow ( check by: net_out.min())
            clip_grad_norm_(parameters=net.parameters(), max_norm=1.0)
            # update network with new adjusted parameters
            optimizer.step()

        loss_epoch[e] = loss.item()
    y_pred = test_predict(net, arg, x_test)
    result = triage_scores(y_pred, y_test, arg)
    return result


if __name__ == "__main__":
    import os

    # Read From
    LDU_train_data_dir = 'outputs/LDU_train_data.csv.gz'
    LDU_test_data_dir = 'outputs/LDU_test_data.csv.gz'
    # Save to
    result_file = 'outputs/score_for_learning_to_defer_with_uncertainty.csv'

    train_data = pd.read_csv(LDU_train_data_dir, compression='gzip', sep=',')
    test_data = pd.read_csv(LDU_test_data_dir, compression='gzip', sep=',')

    # Convert testing data to tensors
    testNP = test_data.drop('label', axis=1).to_numpy()
    test_labelNP = test_data.label.to_numpy()
    x_test = torch.FloatTensor(testNP.tolist())
    y_test = torch.LongTensor(test_labelNP.tolist())
    print(x_test.shape, y_test.shape)

    # Convert training data to tensors
    trainNP = train_data.drop('label', axis=1).to_numpy()
    train_labelNP = train_data.label.to_numpy()
    x = torch.FloatTensor(trainNP.tolist())
    y = torch.LongTensor(train_labelNP.tolist())
    print(x.shape, y.shape)

    # Training parameters
    arg = dict(
        input_size=x_test.shape[1],  # Size of the input layer n_ensembles + 2
        output_size=2,  # Binary prediction
        hidden_size=200,  # Size of the hidden layer
        epochs=20,
        evaluation_interval=20,
        batch_size=50,  # size of mini batches during training
        learning_rate=0.00005,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_ensembles=50,
        seed=47,
        train_size=9600,
        test_size=4160,
        n_classes=2
    )

    loss_model = nn.CrossEntropyLoss().to(arg['device'])
    loss_expert = nn.CrossEntropyLoss().to(arg['device'])

    df_score = pd.DataFrame(columns=['Alpha', 'Deferred_Ratio', 'Deferred_Size', 'Auto_Size', \
                                     'Overall_F1', 'Overall_Accuracy', 'Overall_Sensitivity', 'Overall_Specificity', \
                                     'Model_F1', 'Model_Accuracy', 'Model_Sensitivity', 'Model_Specificity'],
                            dtype=object)

    alpha_tune = np.arange(1.07, 1.4, 0.01).tolist()
    i = 0
    for alpha in alpha_tune:
        result = train(x, y, x_test, y_test, arg, alpha)
        df_score.loc[i] = [alpha] + result
        i += 1

    print(df_score)
    df_score.to_csv(result_file,  index=False)