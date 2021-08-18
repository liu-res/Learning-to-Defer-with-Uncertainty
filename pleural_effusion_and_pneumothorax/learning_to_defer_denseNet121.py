#!~/.venvs/pt/bin/python
# -*- coding: utf-8 -*-
# This code uses the source in reference : https://github.com/gaetandi/cheXpert

import os
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import random
import subprocess
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Train parameters: batch size, maximum number of epochs, etc.
# The network was trained over 2 epochs for the diagnosis of pleural effusion prediction, and over 4 epochs for the diagnosis of pneumothorax
arg = dict(
    trBatchSize=16,
    trMaxEpoch=4,
    nnClassCount=2,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold=0.5,
    rand_seed=47,
    imgtransCrop=224
)

class DenseNet121_CE(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, out_size):
        super(DenseNet121_CE, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, arg['nnClassCount']+1)  # for CE use 2, for LCE use 3
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def loss_function(outputs, labels, n_classes, alpha):
    # Assume expert's predictions always correct and the expert cost is a constant
    # labels: ground truth, n_classes: number of classes
    batch_size = outputs.size()[0]
    out_softmax = torch.nn.functional.softmax(outputs[range(batch_size), :], dim=1)

    # defer class:rc
    rc = [n_classes] * batch_size
    rc = torch.tensor(rc).to(arg['device'])
    loss_rc = -torch.log2(out_softmax[range(batch_size), rc[range(batch_size)].long()])
    loss_rc = torch.sum(loss_rc) / (batch_size + 0.000001)

    loss_class = -torch.log2(out_softmax[range(batch_size), labels[range(batch_size)].long()])
    loss_class = torch.sum(loss_class) / (batch_size + 0.000001)

    loss = loss_rc + alpha * loss_class
    return loss


class CheXpertTrainer_CE():

    def train(seed, model, dataLoaderTrain, trMaxEpoch, n_classes, alpha, checkpoint):

        # SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode='min')

        # SETTINGS: LOSS
        loss = 0

        # LOAD CHECKPOINT
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # TRAIN THE NETWORK
        lossMIN = 100000

        for epochID in range(0, trMaxEpoch):
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            seed_epoch = seed + epochID * 4000* 2
            batchs, losst = CheXpertTrainer_CE.epochTrain(seed_epoch, model, dataLoaderTrain, optimizer, n_classes, alpha)

        return model
        # --------------------------------------------------------------------------------

    def epochTrain(seed, model, dataLoader, optimizer, n_classes, alpha):
        batch = []
        losstrain = []

        # torch.manual_seed(seed)
        model.train()

        for batchID, (varInput, target) in enumerate(dataLoader):
            target = target.cuda(non_blocking=True)
            # set random seed for replication
            random.seed(seed + batchID)
            torch.manual_seed(seed + batchID)

            varOutput = model(varInput)
            lossvalue = loss_function(varOutput, target, n_classes, alpha)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

            l = lossvalue.item()
            losstrain.append(l)

        # Fill three arrays to see the evolution of the loss
        batch.append(batchID)

        return batch, losstrain

    # --------------------------------------------------------------------------------

    def epochVal(model, dataLoader, n_classes, alpha):

        model.eval()

        lossVal = 0
        lossValNorm = 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoader):
                target = target.cuda(non_blocking=True)
                varOutput = model(varInput)

                losstensor = loss_function(varOutput, target, n_classes, alpha)
                lossVal += losstensor
                lossValNorm += 1

        outLoss = lossVal / lossValNorm
        return outLoss


def test_model(model, dataloader):
    model.eval()
    prob_y = []
    pred_y = []
    true_y = []
    # get predictions
    with torch.no_grad():
        for i, (varInput, target) in enumerate(dataloader):
            target = target.cuda(non_blocking=True)
            out = model(varInput)
            out_softmax_class = torch.nn.functional.softmax(out, dim=1)

            pred_y.extend(torch.max(out_softmax_class.cpu().data, 1)[1].numpy().astype(int))
            prob_y.extend(out_softmax_class.cpu().data.numpy()[:, 1].tolist())
            true_y.extend(target.cpu().data.numpy())
    return prob_y, pred_y, true_y


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


def calculate_scores(y_pred, y_true):
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


# dataset
class get_dataset(Dataset):
    def __init__(self, df_file, transform=None):
        self.image_names = df_file.path.tolist()
        self.labels = df_file.label.tolist()
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]

        if os.path.isfile(image_name) and (os.stat(image_name).st_size != 0):
            image = Image.open(image_name).convert('RGB')
            label = self.labels[index]
            # set random seed for replication
            random.seed(arg['seed']+index)
            torch.manual_seed(arg['seed']+index)
            if self.transform is not None:
                image = self.transform(image)
            return image, torch.as_tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Randomization Index')
    parser.add_argument('-seedId', dest='seedId', type=int, help='seed index')
    parser.add_argument('-rid', dest='rid', type=int, help='alpha index')
    arguments = parser.parse_args()
    print('seed index: ', arguments.seedId)

    train_file = 'Dir_to_Train_Dataset.csv.gz'  # The csv files for training dataset, containing 'label' and 'path' of images
    test_file = 'Dir_to_Test_Dataset.csv.gz'  # The csv files for testing dataset, containing 'label' and 'path' of images
    df_train = pd.read_csv(train_file, compression='gzip', sep=',')
    df_test = pd.read_csv(test_file, compression='gzip', sep=',')

    # fix_dir = 'image_folder_on_server/'
    # df_train['path'] = df_train['path'].apply(lambda x: (fix_dir + x.split('/')[-1]))
    # df_test['path'] = df_test['path'].apply(lambda x: (fix_dir + x.split('/')[-1]))

    # Seeds for ensembles models for replication:
    random.seed(arg['rand_seed'])
    model_seeds = random.sample(range(1000), 50)
    # Set seed for this model:
    seed_i = model_seeds[arguments.seedId]
    arg['seed'] = seed_i

    # Alpha range:
    alpha_tune = np.arange(0.7, 1.7, 0.1).tolist()
    alpha = alpha_tune[arguments.rid]

    # Initialize Transforms:
    torch.manual_seed(seed_i)
    random.seed(seed_i)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    torch.manual_seed(seed_i)
    random.seed(seed_i)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(arg['imgtransCrop']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])
    torch.manual_seed(seed_i)
    random.seed(seed_i)
    transform_test = transforms.Compose(
        [transforms.Resize((arg['imgtransCrop'], arg['imgtransCrop'])), transforms.ToTensor(), normalize])

    # Form Dataset:
    dataset_train = get_dataset(df_train, transform_train)
    dataset_test = get_dataset(df_test, transform_test)
    print('train: ', len(dataset_train), ' test: ', len(dataset_test))

    # Initialize model
    random.seed(seed_i)
    torch.manual_seed(seed_i)

    model_classifier = DenseNet121_CE(arg['nnClassCount']).cuda()
    model_classifier = torch.nn.DataParallel(model_classifier).cuda()

    # Data Loader
    random.seed(seed_i)
    torch.manual_seed(seed_i)

    def seed_worker(worker_id):
        np.random.seed(seed_i + worker_id)
        random.seed(seed_i + worker_id)

    dataLoaderTrain = DataLoader(dataset=dataset_train, batch_size=arg['trBatchSize'], shuffle=True, num_workers=16,
                                 pin_memory=True, worker_init_fn=seed_worker)
    dataLoaderTest = DataLoader(dataset=dataset_test, batch_size=arg['trBatchSize'], shuffle=False, num_workers=16,
                                pin_memory=True, worker_init_fn=seed_worker)

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    model = CheXpertTrainer_CE.train(seed_i, model_classifier, dataLoaderTrain, \
                                     arg['trMaxEpoch'], arg['n_classes'], alpha, \
                                     checkpoint=None)

    prob_y, pred_y, true_y = test_model(model, dataLoaderTest)

    score_list = calculate_scores(pred_y, true_y)

    # save the score
    result_path = 'outputs'
    score_file = 'score_of_learning_to_defer_{}.csv'.format(str(alpha))

    df_score = pd.DataFrame(columns=['Alpha', 'Deferred_Ratio', 'Deferred_Size', 'Auto_Size', \
                                     'Overall_F1', 'Overall_Accuracy', 'Overall_Sensitivity', 'Overall_Specificity', \
                                     'Model_F1', 'Model_Accuracy', 'Model_Sensitivity', 'Model_Specificity'],
                            dtype=object)
    df_score.loc[0] = [alpha] + score_list
    df_score.to_csv(os.path.join(result_path, score_file), index=False)
