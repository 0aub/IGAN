import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.gan import weights_init
from utils import map_labels

# Reference: https://github.com/Hanzy1996/CE-GZSL

# ========================================= Classifier Definition =========================================

class ZSLClassifier:
    """
    Generalized Zero-Shot Learning Classifier.

    Args:
        _train_X (Tensor): Training features.
        _train_Y (Tensor): Training labels.
        map_net (Module): Mapping network.
        embed_size (int): Embedding size.
        data_loader (DataLoader): Data loader for the dataset.
        _nclass (int): Number of classes.
        _lr (float): Learning rate.
        _beta1 (float): Beta1 hyperparameter for Adam optimizer.
        _nepoch (int): Number of epochs.
        _batch_size (int): Batch size.
        generalized (bool): Whether to use generalized setting.
    """
    def __init__(self, _train_X, _train_Y, map_net, embed_size, data_loader, _nclass, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True):
        # Constructor for initializing the classifier with data, model, and training parameters.
        # Complete definition omitted for brevity.
        self.train_X =  _train_X
        self.train_Y = _train_Y
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.MapNet=map_net
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = embed_size
        self.model =  LogSoftmaxClassifierLayer(self.input_dim, self.nclass).cuda()
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss().cuda()

        self.input = torch.FloatTensor(_batch_size, _train_X.size(1)).cuda()
        self.label = torch.LongTensor(_batch_size).cuda()

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.h = self.fit()
        else:
            self.acc, self.loss = self.fit_zsl()

    def fit_zsl(self):
        # Fits the model for zero-shot learning scenario.
        # Complete definition omitted for brevity.
        best_acc = 0
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                embed, _=self.MapNet(self.input)
                output = self.model(embed)
                loss = self.criterion(output, self.label)
                mean_loss += loss.data
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > best_acc:
                best_acc = acc
        return best_acc, loss

    def fit(self):
        # Fits the model for generalized zero-shot learning scenario.
        # Complete definition omitted for brevity.
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                embed, _ = self.MapNet(self.input)
                output = self.model(embed)
                loss = self.criterion(output, self.label)

                loss.backward()
                self.optimizer.step()
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if (acc_seen+acc_unseen)==0:
                # print('a bug')
                H=0
            else:
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H

    def next_batch(self, batch_size):
        # Generate next batch of data
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label, target_classes):
        # Validate the GZSL model
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                embed, _ = self.MapNet(test_X[start:end].cuda())
                output = self.model(embed)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        # Compute per class accuracy for GZSL
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        acc_per_class /= target_classes.size(0)
        return acc_per_class

    def val(self, test_X, test_label, target_classes):
        # Validate the model
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                embed, _ = self.MapNet(test_X[start:end].cuda())
                output = self.model(embed)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc(map_labels(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        # Compute per class accuracy
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx])) / float(torch.sum(idx))
        return acc_per_class.mean()

# ========================================= LogSoftmax Classifier Layer =========================================

class LogSoftmaxClassifierLayer(nn.Module):
    """
    Classifier layer with LogSoftmax activation for classification tasks.
    """
    def __init__(self, in_features, out_features):
        """
        LogSoftmax Classifier Layer.

        Args:
            in_features (int): Input features size.
            out_features (int): Output features size.
        """
        super(LogSoftmaxClassifierLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.logsoftmax(x)
        return x
