# -*- coding: utf-8 -*-

import argparse
from dataset import HMDataset
from dataloader import BertDataloader
from models import BERTModel
from loggers import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim

import json
from tqdm import tqdm
from pprint import pprint
import pdb

def parse_args():

    parser = argparse.ArgumentParser(description='kaggle recommender system')

    parser.add_argument('--dataset_path' ,'-dp', type=str, default='/home/mnt/SoundsGood/hm_kaggle/')
    parser.add_argument('--min_tran_len', '-mt', type=int, default=5)
    parser.add_argument('--min_item_count','-mi', type=int, default=0)
    parser.add_argument('--remove_duplicate','-rd', type=bool, default=False)
    parser.add_argument('--num_recent_months', type=int, default=12)
    parser.add_argument('--dataloader_random_seed', '-dlseed', type=int, default=123)
    parser.add_argument('--max_len', '-ml', type=int, default=100)
    parser.add_argument('--mask_prob', '-mp', type=float, default=0.2)
    parser.add_argument('--test_negative_sampler', '-ns', type=str, default='popular', choices=['popular','random'])
    parser.add_argument('--test_negative_sample_size', type=int, default=100)
    parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

    #BERT
    parser.add_argument('--bert_num_blocks', default=2)
    parser.add_argument('--bert_num_heads', default=4)
    parser.add_argument('--bert_hidden_units', default=256)
    parser.add_argument('--bert_dropout', default=0.1)

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_gpu', type=int, default=1)

    parser.add_argument('--log_period_as_iter', type=int, default=12800)

    parser.add_argument('--experiment_dir', type=str, default='experiments')
    parser.add_argument('--experiment_description', type=str, default='test')
    parser.add_argument('--test', type=bool, default=False)

    return parser.parse_args()

def main():

    args = parse_args()
    dataset = HMDataset(args)
    dataloader = BertDataloader(args, dataset)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

    #model 정의
    model = BERTModel(args).to(args.device)
    #loss 정의
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    #optimizer & LR 정의
    optimizer =  optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    pprint(args)

    if not args.test:
        # create experiment foloder
        exp_root = create_experiment_export_folder(args)
        # writer, logger정의
        writer, train_loggers, val_loggers = create_loggers(exp_root)
        logger_service = LoggerService(train_loggers, val_loggers)

        accum_iter = 0
        #temporal
        for epoch in range(args.num_epochs):
            model.train()
            average_meter_set = AverageMeterSet()
            tqdm_train_dataloader = tqdm(train_loader)
            for batch_idx, batch in enumerate(tqdm_train_dataloader):
                batch_size = batch[0].size(0)
                batch = [x.to(args.device) for x in batch]

                seqs, labels = batch
                logits = model(seqs)
                logits = logits.view(-1, logits.size(-1))  # (B*T) x V
                labels = labels.view(-1) # B*T
                loss = ce_loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                average_meter_set.update('loss', loss.item())
                tqdm_train_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch+1,
                                                                                      average_meter_set['loss'].avg))

                accum_iter += batch_size
                needs_to_log = accum_iter % args.log_period_as_iter < args.train_batch_size and accum_iter != 0
                if needs_to_log :
                    tqdm_train_dataloader.set_description('Logging to Tensorboard')
                    state_dict ={'model_state_dict' : model.state_dict(),
                                 'optimizer_state_dict' : optimizer.state_dict()}
                    log_data = {'state_dict' : (state_dict),
                                'epoch': epoch+1,
                                'accum_iter': accum_iter}
                    log_data.update(average_meter_set.averages())
                    logger_service.log_train(log_data)

            lr_scheduler.step()

            #validate
            model.eval()
            average_meter_set = AverageMeterSet()
            with torch.no_grad():
                tqdm_val_dataloader = tqdm(val_loader)
                for batch_idx, batch in enumerate(tqdm_val_dataloader):
                    batch = [x.to(args.device) for x in batch]

                    seqs, candidates, labels = batch
                    scores = model(seqs)
                    scores = scores[:, -1, :]  # B x V # Last Time, all items
                    scores = scores.gather(1, candidates)  # B x C #candidate에 해당되는 것만

                    map = map_at_k(labels, scores, k=12)
                    average_meter_set.update('MAP@12', map)
                    tqdm_val_dataloader.set_description('Epoch {}, '
                                                        'Val - MAP@12 {:.3f}'.format(epoch+1,
                                                                                 average_meter_set['map@12'].avg))

                state_dict = {'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                log_data = {
                    'state_dict': (state_dict),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                logger_service.log_val(log_data)

        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        logger_service.complete({'state_dict':(state_dict)})

    else: # test

        print('Test best model with test set!')
        print(os.listdir('./experiments'))
        test_folder_path = os.path.join('./experiments',input('Choose Test Folder for the test : '))
        best_model = torch.load(os.path.join(test_folder_path, 'models', 'best_acc_model.pth')).get('model_state_dict')
        model.load_state_dict(best_model)
        model.eval()

        average_meter_set = AverageMeterSet()
        c = 0
        with torch.no_grad():
            tqdm_test_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_test_dataloader):
                batch = [x.to(args.device) for x in batch]

                seqs, candidates, labels = batch
                scores = model(seqs)
                scores = scores[:, -1, :]  # B x V # Last Time, all items
                scores = scores.gather(1, candidates)  # B x C #candidate에 해당되는 것만

                map = map_at_k(labels, scores, k=12)
                average_meter_set.update('MAP@12', map)
                tqdm_test_dataloader.set_description('Test -'
                                                     ' MAP@12 {:.3f}'.format(average_meter_set['map@12'].avg))
                c+=1
                if c == 30:
                    break

            average_metrics = average_meter_set.averages()
            with open(os.path.join(test_folder_path, 'logs','test_metrics.json'),'w') as f:
                json.dump(average_metrics, f)
            print(average_metrics)

if __name__ == '__main__':
    main()



