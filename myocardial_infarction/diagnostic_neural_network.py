#!~/.venvs/pt/bin/python
# -*- coding: utf-8 -*-
# This code uses sources listed in the reference below:
# Reference:
# [1] Andriy Mulyar, Elliot Schumacher, Masoud Rouhizadeh, and Mark Dredze. 2019. Phenotyping of Clinical Notes with Improved Document Classification Models Using Contextualized Neural Language Models. arXiv preprint arXiv:1910.13664 (BERT Long Document Classification github repository: https://github.com/AndriyMulyar/bert_document_classification/blob/e9d9cd4dc810630f05661f777923632e3d8fe097/bert_document_classification/document_bert.py)
# [2] Emily Alsentzer, John R. Murphy, Willie Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, Matthew B. A. McDermott, 2019, Publicly Available Clinical BERT Embeddings (https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT)

import math
import torch
import transformers as ppb
import random
from torch import nn
from torch.nn import LSTM
from torch.nn.utils import clip_grad_norm_
import transformers as ppb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import os


def encode_documents(documents, tokenizer_path, max_input_length=512):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
    dimension and the others encode bert input.
    This is the input to any of the document bert architectures.
    :param documents: a list of text documents
    :param tokenizer: the sentence piece bert tokenizer
    :return:
    """
    tokenizer = ppb.AutoTokenizer.from_pretrained(tokenizer_path)
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    max_sequences_per_document = math.ceil(max(len(x) / (max_input_length - 2) for x in tokenized_documents))
    # assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"

    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, 512), dtype=torch.long)
    document_seq_lengths = []  # number of sequence generated per document
    # Need to use 510 to account for 2 padding tokens
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length - 2))):
            raw_tokens = tokenized_document[i:i + (max_input_length - 2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == 512 and len(attention_masks) == 512 and len(input_type_ids) == 512

            # we are ready to rumble
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                      torch.LongTensor(input_type_ids).unsqueeze(0),
                                                      torch.LongTensor(attention_masks).unsqueeze(0)),
                                                     dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index + 1)
    return output, torch.LongTensor(document_seq_lengths)


def gen_last_hidden_layer(df_data, args, tag, i, seed_i):
    # Initialize the model by random seed
    torch.cuda.empty_cache()
    torch.manual_seed(seed_i)
    model_config = ppb.AutoConfig.from_pretrained(args['model_path'], output_hidden_states=True,
                                                  output_all_encoded_layers=True)
    bert = ppb.AutoModel.from_pretrained(args['model_path'], config=model_config).to(args['device'])

    # Freeze bert encoder of all layers:
    for param in bert.parameters():
        param.requires_grad = False

    # Unfreeze bert pooler:
    for name, param in bert.named_parameters():
        if "pooler" in name:
            param.requires_grad = True

    bert.pooler.requires_grad_ = True

    # get tokenized for the chunk
    df_ = df_data.iloc[i * args['chunk_size']: (i + 1) * args['chunk_size']]
    documents = df_['TEXT'].tolist()
    tokenized_block, sequence_lengths = encode_documents(documents, args['tokenizer_path'])
    tokenized_block.to(args['device'])

    last_hidden_state = torch.zeros(size=(tokenized_block.shape[0],
                                          min(tokenized_block.shape[1], args['bert_seq_num']),
                                          args['bert_tok_num'],
                                          bert.config.hidden_size), dtype=torch.float, device=args['device'])

    # pass through "document_batch.shape[1]" numbers of sequences into bert.
    with torch.set_grad_enabled(False):
        for doc_id in range(tokenized_block.shape[0]):
            torch.manual_seed(seed_i + doc_id + i * args['chunk_size'])
            last_hidden_state[doc_id][:args['bert_seq_num']] = \
                bert(tokenized_block[doc_id][:args['bert_seq_num'], 0].to(args['device']),
                     token_type_ids=tokenized_block[doc_id][:args['bert_seq_num'], 1].to(args['device']),
                     attention_mask=tokenized_block[doc_id][:args['bert_seq_num'], 2].to(args['device']))[0]

    label_NP = df_.LABEL.to_numpy()
    label_tensor = torch.LongTensor(label_NP.tolist())

    return last_hidden_state, label_tensor


class DocumentBertLSTM(nn.Module):
    """
    BERT output over document in LSTM
    """

    def __init__(self, model_path):
        super(DocumentBertLSTM, self).__init__()
        self.model_config = ppb.AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        self.bert_seq_num = 13
        self.bert_tok_num = 512
        self.bert = ppb.AutoModel.from_pretrained(model_path, config=self.model_config)
        self.dropout = nn.Dropout(p=self.model_config.hidden_dropout_prob).cuda()
        self.lstm = LSTM(self.model_config.hidden_size, self.model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.model_config.hidden_dropout_prob),
            nn.Linear(self.model_config.hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, last_hidden_state, device):
        bert_pooled_output = torch.zeros(size=(last_hidden_state.shape[0],
                                               last_hidden_state.shape[1],
                                               self.bert.config.hidden_size), dtype=torch.float, device=device)

        for doc_id in range(last_hidden_state.shape[0]):
            bert_pooled_output[doc_id] = self.dropout(self.bert.pooler(last_hidden_state[doc_id]))

        # Re-arrange document embeddings dimension: seq_length x batch_size x input_size, as LSTM input:
        output, (_, _) = self.lstm(bert_pooled_output.permute(1, 0, 2))
        # use the last output
        last_layer = output[-1]

        # prediction are probability of positive class
        confidences = self.classifier(last_layer)
        assert confidences.shape[0] == last_hidden_state.shape[0]
        return confidences

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


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


def predict(model, df_test, args, seed_i):
    with torch.set_grad_enabled(False):
        # predictions on test set:
        test_preds = np.empty(0).astype(int)
        test_probs = np.empty(0).astype(float)
        test_labels = []
        for i in list(range(args['test_chunks'])):
            test_dataloader = get_dataloader(df_test, i, args, shuffle=False, tag='test', seed_i=seed_i)
            for _, test_sample in enumerate(test_dataloader):
                test_tensor_batch = test_sample[0].to(args['device'])
                test_label_batch = test_sample[1].to(args['device'])
                out_mini = model(test_tensor_batch, device=args['device'])

                out_softmax = torch.nn.functional.softmax(out_mini, dim=1)

                test_labels.extend(test_label_batch.reshape(-1).tolist())
                test_preds = np.concatenate([test_preds, torch.max(out_softmax.cpu().data, 1)[1].numpy().astype(int)])
                test_probs = np.concatenate((test_probs, out_softmax.cpu().data.numpy()[:, 1]), axis=0)

            del test_dataloader
            del test_tensor_batch
            del test_label_batch
            del out_mini

        # test scores:
        score_list = get_scores(test_labels, test_preds.tolist())
        return score_list, test_probs.tolist(), test_preds.tolist(), test_labels


def get_dataloader(df_, i, args, shuffle, tag, seed_i):
    last_hidden_state, label_tensor = gen_last_hidden_layer(df_, args, tag, i, seed_i)
    # Combine the tokenized representations and labels into a TensorDataset.
    dataset = TensorDataset(last_hidden_state, label_tensor)
    # shuffle=True for train dataloader, False for testing dataloader
    np.random.seed(seed_i)
    random.seed(seed_i)
    torch.manual_seed(seed_i)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=args['batch_size'])
    return dataloader


def train(df_train, df_test, args, seed_i):
    torch.cuda.empty_cache()
    torch.manual_seed(seed_i)
    model = DocumentBertLSTM(args['model_path'])
    model.to(args['device'])

    model.freeze_bert_encoder()
    model.unfreeze_bert_encoder_pooler_layer()

    loss_function = nn.CrossEntropyLoss().to(args['device'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        weight_decay=args['weight_decay'],
        lr=args['learning_rate']
    )

    # Start training
    for i in list(range(args['train_chunks'])):
        train_dataloader = get_dataloader(df_train, i, args, shuffle=True, tag='train', seed_i=seed_i)

        for epoch in range(args['max_epoch']):
            num_batch = 0
            for batch_ndx, sample in enumerate(train_dataloader):
                num_batch += 1
                batch_document_tensors = sample[0].to(args['device'])  # Tensor: batch_size * tokenized
                batch_correct_output = sample[1].to(args['device'])  # Tensor: batch_size * label

                # set random seed for replication
                torch.manual_seed(seed_i + i * args['max_epoch'] * 10 + epoch * 10 + num_batch)
                batch_confidences = model(batch_document_tensors, device=args['device'])

                loss = loss_function(batch_confidences, batch_correct_output)

                model.zero_grad()
                loss.backward()

                clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

    # Evaluation at max epoch
    score_list, y_prob, y_pred, y_ture = predict(model, df_test, args, seed_i)

    return score_list, y_prob, y_pred, y_ture


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Raw data dir')
    parser.add_argument('-train', dest='train', type=str, help="Dir of train dataframe with 'LABEL' and 'TEXT' columns")
    parser.add_argument('-test', dest='test', type=str, help="Dir of test dataframe with 'LABEL' and 'TEXT' columns")
    parser.add_argument('-sid', dest='sid', type=int, help='Random seed index')
    arguments = parser.parse_args()
    print('sid: ', arguments.sid)
    sid = arguments.sid

    args = dict(batch_size=16,
                max_epoch=18,
                learning_rate=1e-4,
                weight_decay=0,
                model_path="emilyalsentzer/Bio_Discharge_Summary_BERT",
                tokenizer_path="emilyalsentzer/Bio_Discharge_Summary_BERT",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                evaluation_interval=18,
                n_ensembles=50,
                seed=47,
                bert_seq_num=10,
                bert_tok_num=512,
                train_size=9645,
                test_size=4160,
                chunk_size=160,
                train_chunks=math.ceil(9645 / 160),
                test_chunks=math.ceil(4160 / 160),
                threshold=0.5,
                n_classes=2
                )

    df_train = pd.read_csv(arguments.train, compression='gzip', sep=',')
    df_test = pd.read_csv(arguments.test, compression='gzip', sep=',')
    print('train size: ', df_train.shape[0], 'test size: ', df_test.shape[0])

    # Generate random seed list for replication
    random.seed(args['seed'])
    list_seed = random.sample(range(100), args['n_ensembles'])
    seed_i = list_seed[sid]

    score_list, y_prob, y_pred, y_true = train(df_train, df_test, args, seed_i)
    df_base = pd.DataFrame([y_pred, y_prob, y_true], index=['pred', 'prob', 'label']).T
    df_score = pd.DataFrame(columns=['Model_F1', 'Model_Accuracy', 'Model_Sensitivity', 'Model_Specificity'],
                            dtype=object)
    df_score.loc[0] = score_list

    # Store the result depending on whether the model is evaluated on training dataset(for LDU input) or test dataset:
    # output_train{}.csv.gz, or output_test{}.csv.gz
    # score_train{}.csv or score_test{}.csv
    df_base.to_csv('outputs/output_test{}.csv.gz'.format(sid), index=False, compression='gzip')
    df_score.to_csv('outputs/score_test{}.csv'.format(sid), index=False)