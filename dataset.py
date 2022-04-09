# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from pathlib import Path
import os
import pickle
import pdb

class HMDataset:

    def __init__(self, args):

        self.dataset_path = args.dataset_path
        self.min_tran_len = args.min_tran_len
        self.min_item_count = args.min_item_count
        self.remove_duplicate = args.remove_duplicate
        self.num_recent_months = args.num_recent_months

    def load_dataset(self):
        self.preprocess()
        prep_dataset_path = self._get_preprocessed_dataset_path()
        with prep_dataset_path.open('rb') as f:
            dataset = pickle.load(f)
        return dataset

    def preprocess(self):
        prep_dataset_path = self._get_preprocessed_dataset_path()
        if prep_dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        tran_df = self.load_transaction_df()
        tran_df = self.filter_transaction(tran_df)
        df, cmap, amap = self.densify_index(tran_df)
        train, val, test = self.split_df(df, len(cmap))
        dataset = {'train' : train,
                   'val' : val,
                   'test' : test,
                   'cmap' : cmap,
                   'amap' : amap}
        with prep_dataset_path.open('wb') as f:
            pickle.dump(dataset, f)


    def load_transaction_df(self):
        print('loading transaction...')
        transaction_path = os.path.join(self.dataset_path, 'transactions_train.csv')
        df = pd.read_csv(transaction_path)
        df['t_dat'] = pd.to_datetime(df['t_dat'])
        return df

    def filter_transaction(self, df):
        print('Filtering transaction...')
        if self.min_item_count > 0:
            """추후에 필요하면 기능 추가."""
            """transaction 내 거래량 적은 article들 제거."""
            pass
        if self.remove_duplicate:
            """추후에 필요하면 기능 추가."""
            """중복해서 구매된 제품 한번 구매된 걸로 counting."""
            pass

        if self.num_recent_months > 0:
            most_recent_date = df['t_dat'].max()
            previous_date = most_recent_date - pd.DateOffset(months = self.num_recent_months)
            df = df.loc[(df['t_dat'] >= pd.to_datetime('{}-{}-01'.format(previous_date.year,
                                                                    previous_date.month)))]
        if self.min_tran_len > 0:
            tran_sizes = df.groupby(['customer_id']).size()
            good_customers = tran_sizes.index[tran_sizes >= self.min_tran_len]
            df = df[df['customer_id'].isin(good_customers)]
        return df

    def densify_index(self, df):
        print('Densifying index...')
        cmap = {c : i for i, c in enumerate(set(df['customer_id']))}
        amap = {a : i+1 for i, a in enumerate(set(df['article_id']))} #1부터 index 시작 # 0:zero-pad #마지막+1:mask token
        df['customer_id'] = df['customer_id'].map(cmap)
        df['article_id'] = df['article_id'].map(amap)
        return df, cmap, amap

    def split_df(self, df, customer_count):
        print('Splitting...')
        customer_group = df.groupby(['customer_id'])
        customer2article = customer_group.progress_apply(lambda d : list(d.sort_values(by='t_dat')['article_id']))
        train, val, test = {}, {}, {}
        for customer in range(customer_count):
            article = customer2article[customer]
            train[customer], val[customer], test[customer] = article[:-2], article[-2:-1], article[-1:]
        return train, val, test

    def _get_rawdata_root_path(self):
        return Path(self.dataset_path)

    def _get_preprocessed_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_dataset_folder_path(self):
        preprocessed_root = self._get_preprocessed_folder_path()
        if not os.path.exists(preprocessed_root):
            os.makedirs(preprocessed_root)
        pkl_folder = 'min_tran{}-min_item{}-remove_dupli{}-recent_mth{}'.format(self.min_tran_len, self.min_item_count,
                                                                                int(self.remove_duplicate), self.num_recent_months)
        return preprocessed_root.joinpath(pkl_folder)

    def _get_preprocessed_dataset_path(self):
        dataset_root = self._get_preprocessed_dataset_folder_path()
        if not os.path.exists(dataset_root):
            os.makedirs(dataset_root)
        pkl_name = 'dataset.pkl'
        return dataset_root.joinpath(pkl_name)

if __name__ == '__main__':
    dataset_path = '/home/mnt/SoundsGood/hm_kaggle/'
    min_tran_length = 5
    min_item_count = 0
    dataset = HMDataset(dataset_path, min_tran_length, min_item_count)
    dataset.preprocess()