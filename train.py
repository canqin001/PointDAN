import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_pointnet import Pointnet_cls
import Model
from dataloader import Modelnet40_data, Shapenet_data, Scannet_data_h5
from torch.autograd import Variable
import time
import numpy as np
import os
import argparse
import pdb
import mmd
# from utils import *
import math
import warnings

from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='scannet')
parser.add_argument('-target', '-t', type=str, help='target dataset', default='modelnet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=200)
parser.add_argument('-models', '-m', type=str, help='alignment model', default='MDA')
parser.add_argument('-lr',type=float, help='learning rate', default=0.0001)
parser.add_argument('-scaler',type=float, help='scaler of learning rate', default=1.)
parser.add_argument('-weight',type=float, help='weight of src loss', default=1.)
parser.add_argument('-datadir',type=str, help='directory of data', default='./dataset/')
parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./logs')
args = parser.parse_args()

if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
    os.makedirs(os.path.join(os.getcwd(), args.tb_log_dir))
writer = SummaryWriter(log_dir=args.tb_log_dir)

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
num_class = 10
dir_root = os.path.join(args.datadir, 'PointDA_data/')

# print(dir_root)
def main():
    print ('Start Training\nInitiliazing\n')
    print('src:', args.source)
    print('tar:', args.target) 

    # Data loading

    data_func={'modelnet': Modelnet40_data, 'scannet': Scannet_data_h5, 'shapenet': Shapenet_data}

    source_train_dataset = data_func[args.source](pc_input_num=1024, status='train', aug=True, pc_root = dir_root + args.source)
    target_train_dataset1 = data_func[args.target](pc_input_num=1024, status='train', aug=True,  pc_root = dir_root + args.target)
    source_test_dataset = data_func[args.source](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.source)
    target_test_dataset1 = data_func[args.target](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.target)

    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_train1 = len(target_train_dataset1)
    num_target_test1 = len(target_test_dataset1)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    target_train_dataloader1 = DataLoader(target_train_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    target_test_dataloader1 = DataLoader(target_test_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    print('num_source_train: {:d}, num_source_test: {:d}, num_target_test1: {:d} '.format(num_source_train, num_source_test, num_target_test1))
    print('batch_size:', BATCH_SIZE)

    # Model

    model = Model.Net_MDA()
    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    remain_epoch=50

    # Optimizer

    params = [{'params':v} for k,v in model.g.named_parameters() if 'pred_offset' not in k]

    optimizer_g = optim.Adam(params, lr=LR, weight_decay=weight_decay)
    lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs+remain_epoch)

    optimizer_c = optim.Adam([{'params':model.c1.parameters()},{'params':model.c2.parameters()}], lr=LR*2,
                             weight_decay=weight_decay)
    lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=args.epochs+remain_epoch)

    optimizer_dis = optim.Adam([{'params':model.g.parameters()},{'params':model.attention_s.parameters()},{'params':model.attention_t.parameters()}], 
        lr=LR*args.scaler, weight_decay=weight_decay)
    lr_schedule_dis = optim.lr_scheduler.CosineAnnealingLR(optimizer_dis, T_max=args.epochs+remain_epoch)


    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by half by every 5 or 10 epochs"""
        if epoch > 0:
            if epoch <= 30:
                lr = args.lr * args.scaler * (0.5 ** (epoch // 5))
            else:
                lr = args.lr * args.scaler * (0.5 ** (epoch // 10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            writer.add_scalar('lr_dis', lr, epoch)

    def discrepancy(out1, out2):
        """discrepancy loss"""
        out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
        return out

    def make_variable(tensor, volatile=False):
        """Convert Tensor to Variable."""
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return Variable(tensor, volatile=volatile)

    best_target_test_acc = 0

    for epoch in range(max_epoch):
        since_e = time.time()
            
        lr_schedule_g.step(epoch=epoch)
        lr_schedule_c.step(epoch=epoch)
        adjust_learning_rate(optimizer_dis, epoch)

        writer.add_scalar('lr_g', lr_schedule_g.get_lr()[0], epoch)
        writer.add_scalar('lr_c', lr_schedule_c.get_lr()[0], epoch)

        model.train()

        loss_total = 0
        loss_adv_total = 0
        loss_node_total = 0
        correct_total = 0
        data_total = 0
        data_t_total = 0
        cons = math.sin((epoch + 1)/max_epoch * math.pi/2 )

        # Training

        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_train_dataloader, target_train_dataloader1)):

            data, label = batch_s
            data_t, label_t = batch_t

            data = data.to(device=device)
            label = label.to(device=device).long()
            data_t = data_t.to(device=device)
            label_t = label_t.to(device=device).long()

            pred_s1,pred_s2 = model(data)
            pred_t1,pred_t2 = model(data_t, constant = cons, adaptation=True)

            # Classification loss

            loss_s1 = criterion(pred_s1, label)
            loss_s2 = criterion(pred_s2, label)

            # Adversarial loss

            loss_adv = - 1 * discrepancy(pred_t1, pred_t2) 

            loss_s = loss_s1  +  loss_s2
            loss = args.weight * loss_s + loss_adv

            loss.backward()
            optimizer_g.step()
            optimizer_c.step()
            optimizer_g.zero_grad()
            optimizer_c.zero_grad()


            # Local Alignment
            
            feat_node_s = model(data, node_adaptation_s=True)
            feat_node_t = model(data_t, node_adaptation_t=True)
            sigma_list = [0.01, 0.1, 1, 10, 100]
            loss_node_adv = 1 * mmd.mix_rbf_mmd2(feat_node_s, feat_node_t, sigma_list)
            loss = loss_node_adv

            loss.backward()
            optimizer_dis.step()
            optimizer_dis.zero_grad()

            loss_total += loss_s.item() * data.size(0)
            loss_adv_total += loss_adv.item() * data.size(0)
            loss_node_total +=  loss_node_adv.item() * data.size(0)
            data_total += data.size(0)
            data_t_total += data_t.size(0)

            if (batch_idx + 1) % 10 == 0:
                print('Train:{} [{} {}/{}  loss_s: {:.4f} \t loss_adv: {:.4f} \t loss_node_adv: {:.4f} \t cons: {:.4f}]'.format(
                epoch, data_total, data_t_total,num_source_train, loss_total/data_total, 
                loss_adv_total/data_total,  loss_node_total/data_total,  cons
                ))

        # Testing

        with torch.no_grad():
            model.eval()
            loss_total = 0
            correct_total = 0
            data_total = 0
            acc_class = torch.zeros(10,1)
            acc_to_class = torch.zeros(10,1)
            acc_to_all_class = torch.zeros(10,10)

            for batch_idx, (data,label) in enumerate(target_test_dataloader1):
                data = data.to(device=device)
                label = label.to(device=device).long()
                pred1, pred2 = model(data)
                output = (pred1 + pred2)/2
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)

                loss_total += loss.item() * data.size(0)
                correct_total += torch.sum(pred == label)
                data_total += data.size(0)

            pred_loss = loss_total/data_total
            pred_acc = correct_total.double()/data_total

            if pred_acc > best_target_test_acc:
                best_target_test_acc = pred_acc
            print ('Target 1:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Target Acc: {:.4f}]'.format(
            epoch, pred_acc, pred_loss, best_target_test_acc
            ))
            writer.add_scalar('accs/target_test_acc', pred_acc, epoch)


        time_pass_e = time.time() - since_e
        print('The {} epoch takes {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 60, time_pass_e % 60))
        print(args)
        print(' ')


if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))

