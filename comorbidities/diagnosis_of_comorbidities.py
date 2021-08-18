from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import random
import numpy as np
import os
import math
from torch.nn.utils import clip_grad_norm_
from scipy.stats import entropy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# define model
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(arg['input_size'], arg['hidden_size'])
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(arg['hidden_size'], arg['output_size'])
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.l3(x)
        x = F.softmax(x, dim=1)
        return x


def test_predict(net, arg, x_val):
    with torch.set_grad_enabled(False):
        # predictions on validation set:
        val_preds = np.empty(0).astype(int)
        val_probs = np.empty(0).astype(float)
        for i in range(0, x_val.shape[0], arg['batch_size']):
            x_mini = x_val[i:i + arg['batch_size']].to(arg['device'])
            out_mini = net(x_mini)
            val_preds = np.concatenate([val_preds, torch.max(out_mini.cpu().data, 1)[1].numpy().astype(int)])
            val_probs = np.concatenate((val_probs, out_mini.cpu().data.numpy()[:, 1]), axis=0)
            del x_mini
            del out_mini
        return val_preds, val_probs


def train(x, y, x_test, y_test, arg, seed):
    # init Network_rej
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    net = Network().to(arg['device'])
    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=arg['learning_rate'])
    loss_func = nn.CrossEntropyLoss().to(arg['device'])

    loss_epoch = np.zeros(arg['epochs'])
    y_pred = np.empty(0).astype(int)
    y_prob = np.empty(0).astype(float)
    # Training loop
    for e in range(arg['epochs']):
        batch_ndx = -1
        for i in range(0, x.shape[0], arg['batch_size']):
            batch_ndx += 1
            x_mini = x[i:i + arg['batch_size']].to(arg['device'])
            y_mini = y[i:i + arg['batch_size']].to(arg['device'])
            # Wrap mini batch input_mnist tensor to allow automatic gradient computation when backpropagating
            x_var = Variable(x_mini)
            y_var = Variable(y_mini)
            # resets the gradient of the optimizer
            optimizer.zero_grad()
            # forward propagation:
            torch.manual_seed(seed + e * 1000 + i + 100)  # for replication
            net_out = net(x_var)
            # computes the loss
            loss = loss_func(net_out, y_var)
            if math.isnan(loss.item()):
                break
            # compute the gradient based on the loss
            loss.backward()
            # clip gradient to prevent underflow ( check by: net_out.min())
            clip_grad_norm_(parameters=net.parameters(), max_norm=1.0)
            # update network with new adjusted parameters
            optimizer.step()

        loss_epoch[e] = loss.item()
        if (e + 1) % arg['evaluation_interval'] == 0:
            y_pred, y_prob = test_predict(net, arg, x_test)
    return y_pred, y_prob, loss_epoch


def get_scores(true_y, pred_y):
    # true_y: list, pred_y: list
    # F1, and from Confusion matrix compute Accuracy, sensitivity(TP/P) and specificity(TN/N)
    if len(true_y) == 1:
        f1 = None
    else:
        f1 = f1_score(true_y, pred_y)

    if (sum(true_y) == len(true_y)) & (true_y == pred_y):
        print('all are positive, and predicted correctly')
        return [f1, 1.0, 1.0, None]
    elif (sum(true_y) == 0) & (true_y == pred_y):
        print('all are negative, and predicted correctly')
        return [f1, 1.0, None, 1.0]
    else:
        cm = confusion_matrix(true_y, pred_y)
        total = sum(sum(cm))
        accuracy = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        return [f1, accuracy, sensitivity, specificity]


def calculate_scores(y_pred, y_true, arg):
    df = pd.DataFrame(
        {'true_y': y_true,
         'pred_y': y_pred
         })
    df_1 = df[df.pred_y != arg['n_classes']]
    df_2 = df[df.pred_y == arg['n_classes']]
    print('Num of deferred: {}, not deferred {}'.format(df_2.shape[0], df_1.shape[0]))
    if not df_2.empty:
        df_expert = df_2.copy()
        df_expert['pred_y'] = df_expert['true_y']
        df_overall = df_1.append(df_expert)
        deferred_size = df_2.shape[0]
    else:
        df_overall = df_1.copy()
        deferred_size = 0

    deferred_ratio = deferred_size / df.shape[0]

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


def create_ens_data(probs_NP, y_test_List):
    col = ['ens{}'.format(i) for i in range(probs_NP.shape[1])]
    df_ens = pd.DataFrame(columns=col + ['label'])
    for sid in range(probs_NP.shape[1]):
        df_ens['ens{}'.format(sid)] = probs_NP[:, sid].tolist()
    df_ens['label'] = y_test_List
    return df_ens


# Calculate information entropy per sample for pred_y_train
# axis=0 calculate entropy along the columns: per sample
# shannon entropy = -sum(pk * log(pk), axis=axis)
# Information entropy for binary classification
def entropy_binary_class(i_preds):
    # i_preds the predictions from ensembles for ith data point
    pk_1 = i_preds.sum() / len(i_preds)
    pk_2 = 1 - pk_1
    pk = np.array([pk_1, pk_2])
    information_entropy = entropy(pk)
    return information_entropy


def add_entropies(df_data):
    d_entropy = []
    e_entropy = []
    for i in range(df_data.shape[0]):
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


class Network_defer(nn.Module):

    def __init__(self):
        super(Network_defer, self).__init__()
        self.l1 = nn.Linear(arg['input_size'], arg['hidden_size'])
        self.sigmoid = nn.Sigmoid()
        self.l3 = nn.Linear(arg['hidden_size'], arg['n_classes'] + 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.l3(x)
        x = F.softmax(x, dim=1)
        return x


def test_predict_defer(net, arg, x_val):
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


def learn_to_defer(x_new, y_new, x_test, y_test, arg, alpha, seed_i):
    # init Network_rej
    torch.cuda.empty_cache()
    torch.manual_seed(seed_i)
    net = Network_defer().to(arg['device'])
    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=arg['learning_rate'])
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
            torch.manual_seed(seed_i + e * 1000 + i + 100)  # for replication
            net_out = net(x_var)
            # computes the loss
            loss = defer_CELoss(net_out, y_var, arg['n_classes'], alpha, arg)
            # compute the gradient based on the loss
            loss.backward()
            # clip gradient to prevent underflow ( check by: net_out.min())
            clip_grad_norm_(parameters=net.parameters(), max_norm=1.0)
            # update network with new adjusted parameters
            optimizer.step()

        loss_epoch[e] = loss.item()
        if (e + 1) % arg['evaluation_interval'] == 0:
            y_pred = test_predict_defer(net, arg, x_test)
            score_list = calculate_scores(y_pred.tolist(), y_test.tolist(), arg)
            return score_list


# Convert train and test dataframes to tensors
train_data = pd.read_csv('outputs/df_train.csv')
test_data = pd.read_csv('outputs/df_test.csv')

testNP = test_data.drop(['MemberID', 'Label'], axis=1).to_numpy()
test_labelNP = test_data.Label.to_numpy()
x_test = torch.FloatTensor(testNP.tolist())
y_test = torch.LongTensor(test_labelNP.tolist())
print(x_test.shape, y_test.shape)

trainNP = train_data.drop(['MemberID', 'Label'], axis=1).to_numpy()
train_labelNP = train_data.Label.to_numpy()
x = torch.FloatTensor(trainNP.tolist())
y = torch.LongTensor(train_labelNP.tolist())
print(x.shape, y.shape)

# Train parameters
arg = dict(
    input_size=x.shape[1],
    output_size=2,
    n_classes=2,
    hidden_size=200,
    epochs=4,
    evaluation_interval=4,
    batch_size=512,
    learning_rate=0.0009,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold=0.5,
    n_ensembles=50,
    seed=17,
    train_size=x.shape[0],
    test_size=x_test.shape[0]
)

random.seed(arg['seed'])
list_seed = random.sample(range(100), arg['n_ensembles'])

# Diagnostic NN
seed = list_seed[0]
y_pred, y_prob, _ = train(x, y, x_test, y_test, arg, seed)
[f1, accuracy, sensitivity, specificity] = get_scores(y_test.tolist(), y_pred.tolist())
print('[f1, accuracy, sensitivity, specificity]', [f1, accuracy, sensitivity, specificity])

# NN ensembles
train_probs = np.zeros([arg['train_size'], arg['n_ensembles']])
test_probs = np.zeros([arg['test_size'], arg['n_ensembles']])
for i in range(arg['n_ensembles']):
    print('ensemble {}:'.format(i))
    seed = list_seed[i]
    _, train_prob, _ = train(x, y, x, y, arg, seed)
    train_probs[:, i] = train_prob
    test_pred, test_prob, _ = train(x, y, x_test, y_test, arg, seed)
    test_probs[:, i] = test_prob
    print('[f1, accuracy, sensitivity, specificity]', get_scores(y_test.tolist(), test_pred.tolist()))

df_train_ens = create_ens_data(train_probs, y.tolist())
df_test_ens = create_ens_data(test_probs, y_test.tolist())

# Add entropies
df_train_ens = add_entropies(df_train_ens)
df_train_ens.to_csv('outputs/df_train_ens.csv.gz', index=False, compression='gzip')
df_test_ens = add_entropies(df_test_ens)
df_test_ens.to_csv('outputs/df_test_ens.csv.gz', index=False, compression='gzip')

# Get distribution of entropies
df_dist = pd.DataFrame(columns=df_test_ens['ensemble_entropy'].describe().index)
df_dist.loc[0] = df_test_ens['ensemble_entropy'].describe().tolist()
df_dist.loc[1] = df_test_ens['diagnostic_entropy'].describe().tolist()
df_dist.to_csv('outputs/distributions_entropies.csv', index=False)

# Direct triage by uncertainty
thresholds = np.arange(round(df_dist['min'][0], 2) - 0.01, round(df_dist['max'][0], 2) + 0.02, 0.01).tolist()
get_entropy_triage(df_test_ens, thresholds, 'ensemble_entropy', 'outputs/triage_by_ensemble_entropy.csv')

if df_dist['max'][1] > 0:
    thresholds = np.arange(0, round(df_dist['max'][1], 2) + 0.02, 0.01).tolist()
    get_entropy_triage(df_test_ens, thresholds, 'diagnostic_entropy', 'outputs/triage_by_diagnostic_entropy.csv')

# learning to defer loss function
loss_model = nn.CrossEntropyLoss().to(arg['device'])
loss_expert = nn.CrossEntropyLoss().to(arg['device'])


def defer_CELoss(outputs, labels, n_classes, alpha, arg):
    # Assume expert's predictions always correct and the expert cost is a consitant
    # labels: ground truth, n_classes: number of classes
    batch_size = outputs.size()[0]  # batch_size
    # reject_class:rc
    rc = [n_classes] * batch_size
    rc = torch.tensor(rc).to(arg['device'])
    loss = loss_expert(outputs[range(batch_size)], rc) + alpha * loss_model(outputs[range(batch_size)], labels)
    return loss


# Learning to defer
df_ltd_score = pd.DataFrame(columns=['Alpha', 'Deferred_Ratio', 'Deferred_Size', 'Auto_Size', \
                                     'Overall_F1', 'Overall_Accuracy', 'Overall_Sensitivity', 'Overall_Specificity', \
                                     'Model_F1', 'Model_Accuracy', 'Model_Sensitivity', 'Model_Specificity'],
                            dtype=object)

seed = list_seed[0]
alpha_tune = np.arange(1.4, 2.45, 0.05).tolist()
i = 0
for alpha in alpha_tune:
    score_list = learn_to_defer(x, y, x_test, y_test, arg, alpha, seed)
    df_ltd_score.loc[i] = [alpha] + score_list
    i += 1
df_ltd_score.to_csv('outputs/learn_to_defer_result.csv', index=False)

# Learning to defer with Uncertainty

# convert dataframes to tensors
trainNP = df_train_ens.drop('label', axis=1).to_numpy()
train_labelNP = df_train_ens.label.to_numpy()
ens_x = torch.FloatTensor(trainNP.tolist())
ens_y = torch.LongTensor(train_labelNP.tolist())

testNP = df_test_ens.drop('label', axis=1).to_numpy()
test_labelNP = df_test_ens.label.to_numpy()
ens_x_test = torch.FloatTensor(testNP.tolist())
ens_y_test = torch.LongTensor(test_labelNP.tolist())

# parameters:
arg = dict(
    input_size=ens_x.shape[1],
    n_classes=2,
    hidden_size=200,
    epochs=20,
    evaluation_interval=20,
    batch_size=512,
    learning_rate=0.0009, # 0.0008
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold=0.5,
    n_ensembles=50,
    seed=157
)

df_ltd_uncertainty_score = pd.DataFrame(columns=['Alpha', 'Deferred_Ratio', 'Deferred_Size', 'Auto_Size', \
                                                 'Overall_F1', 'Overall_Accuracy', 'Overall_Sensitivity',
                                                 'Overall_Specificity', \
                                                 'Model_F1', 'Model_Accuracy', 'Model_Sensitivity',
                                                 'Model_Specificity'],
                                        dtype=object)

seed = list_seed[0]
alpha_tune = np.arange(1.4, 2.45, 0.05).tolist()
i = 0
for alpha in alpha_tune:
    score_list = learn_to_defer(ens_x, ens_y, ens_x_test, ens_y_test, arg, alpha, seed)
    df_ltd_uncertainty_score.loc[i] = [alpha] + score_list
    i += 1

df_ltd_uncertainty_score.to_csv('outputs/learn_to_defer_w_uncertainty_result.csv', index=False)
