import torch
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

# Reference: https://github.com/Hanzy1996/CE-GZSL




class DataLoader(object):
    """
    A class for loading dataset features, labels, and attributes from different formats.
    Args:
        opt: An object with attribute settings for the dataset, including paths and dataset names.
    """
    def __init__(self, opt):
        self.read_matdataset(opt)

    def read_matdataset(self, opt):
        """Read dataset from MATLAB .mat files for non-ImageNet datasets."""
        matcontent = sio.loadmat(opt.dataset_path + "/" + opt.dataset_name + "/" + opt.image_embedding_type + ".mat")
        feature = matcontent['features'].T
        self.all_file = matcontent['image_files']
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataset_path + "/" + opt.dataset_name + "/" + opt.attribute_representation_type + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attributes = torch.from_numpy(matcontent['att'].T).float()
        if not opt.enable_validation:
            self.train_image_file = self.all_file[trainval_loc]
            self.test_seen_image_file = self.all_file[test_seen_loc]
            self.test_unseen_image_file = self.all_file[test_unseen_loc]

            if opt.enable_preprocessing:
                if opt.enable_standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_features = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_features = torch.from_numpy(_train_features).float()
                mx = self.train_features.max()
                self.train_features.mul_(1 / mx)
                self.train_labels = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_features = torch.from_numpy(feature[trainval_loc]).float()
                self.train_labels = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_features = torch.from_numpy(feature[train_loc]).float()
            self.train_labels = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_labels.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_features.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.attributes_seen = self.attributes[self.seenclasses]
        # collect the data of each class
        self.train_samples_class_index = torch.tensor([self.train_labels.eq(i_class).sum().float() for i_class in self.train_class])


    def next_batch(self, batch_size):
        """Fetch the next batch of data."""
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_features[idx]
        batch_label = self.train_labels[idx]
        batch_att = self.attributes[batch_label]
        return batch_feature, batch_label, batch_att
