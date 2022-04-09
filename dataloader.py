# -*- coding: utf-8 -*-

import random
from negative_sampler import *

import torch
import torch.utils.data as data_utils
import copy
import collections
import pdb


MaskedLmInstance = collections.namedtuple('MaskedLmInstance', ['index','label'])

class BertDataloader:

    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_dataset_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.cmap = dataset['cmap']
        self.amap = dataset['amap']
        self.customer_count = len(self.cmap)
        self.article_count = len(self.amap)
        args.num_articles = len(self.amap)

        self.max_len = args.max_len
        self.mask_prob = args.mask_prob
        self.CLOZE_MASK_TOKEN = self.article_count + 1

        test_negative_sampler_code = args.test_negative_sampler
        test_negative_sampler = negative_sampler_factory(test_negative_sampler_code, self.train, self.val, self.test,
                                 self.customer_count, self.article_count,
                                 args.test_negative_sample_size,
                                 args.test_negative_sampling_seed,
                                 self.save_folder)
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        if mode == 'val':
            train = self.train
        else: # mode == 'test'
            train = {}
            for c in self.train.keys():
                train[c] = self.train[c] + self.val[c]
        dataset = BertEvalDataset(train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                  self.test_negative_samples)
        return dataset

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN,
                                   self.article_count, self.rng)
        return dataset

class BertTrainDataset(data_utils.Dataset):

    def __init__(self, c2seq, max_len, mask_prob, mask_token, num_articles, rng):
        self.c2seq = c2seq
        self.customers = sorted(self.c2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_articles = num_articles
        self.rng = rng

    def __len__(self):
        return len(self.customers)

    def __getitem__(self, index):
        customer = self.customers[index]
        seq = self.c2seq[customer]

        tokens = copy.deepcopy(seq)
        labels = [0] * len(seq)
        seq_inds = np.arange(len(seq))
        self.rng.shuffle(seq_inds)

        count_masked_token = 0
        covered_index = set()
        len_masked_token = len(seq) * self.mask_prob

        for ind in seq_inds:
            if count_masked_token >= len_masked_token:
                break
            if ind in covered_index:
                continue
            covered_index.add(ind)
            tokens[ind] = self.mask_token
            labels[ind] = seq[ind]
            count_masked_token += 1

        tokens = tokens[-self.max_len:] # 가장 최근 max_len 길이 만큼만
        labels = labels[-self.max_len:]

        #padding
        padding_len = self.max_len - len(tokens)
        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

class BertEvalDataset(data_utils.Dataset):

    def __init__(self, c2seq, c2answer, max_len, mask_token, negative_samples):
        self.c2seq = c2seq
        self.customers = sorted(self.c2seq.keys())
        self.c2answer = c2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.customers)

    def __getitem__(self, index):
        customer = self.customers[index]
        seq = self.c2seq[customer]
        answer = self.c2answer[customer]
        negs = self.negative_samples[customer]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)








