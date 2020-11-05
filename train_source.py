import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_pointnet import Pointnet_cls
from dataloader import Modelnet40_data, Shapenet_data, Scannet_data_h5
import time
import os
import argparse

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='scannet')
parser.add_argument('-target1', '-t1', type=str, help='target dataset', default='modelnet')
parser.add_argument('-target2', '-t2', type=str, help='target dataset', default='shapenet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=200)
parser.add_argument('-lr', type=float, help='learning rate', default=0.001)
parser.add_argument('-datadir', type=str, help='directory of data', default='/repository/yhx/')
parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./logs/src_m_s_ss')
args = parser.parse_args()

if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
    from tensorboardX import SummaryWriter
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


def main():
    """
    Main function.

    Args:
    """
    print ('Start Training\nInitiliazing\n')
    print('src:', args.source)
    print('tar1:', args.target1)
    print('tar2:', args.target2)
    data_func={'modelnet': Modelnet40_data, 'scannet': Scannet_data_h5, 'shapenet': Shapenet_data}

    source_train_dataset = data_func[args.source](pc_input_num=1024, status='train', aug=True, pc_root = dir_root + args.source)
    source_test_dataset = data_func[args.source](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.source)
    target_test_dataset1 = data_func[args.target1](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.target1)
    target_test_dataset2 = data_func[args.target2](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.target2)


    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_test1 = len(target_test_dataset1)
    num_target_test2 = len(target_test_dataset2)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    target_test_dataloader1 = DataLoader(target_test_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    target_test_dataloader2 = DataLoader(target_test_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)

    print('num_source_train: {:d}, num_source_test: {:d}, num_target_test1: {:d}, num_target_test2: {:d}'.format(
        num_source_train, num_source_test, num_target_test1, num_target_test2))
    print('batch_size:', BATCH_SIZE)

    # Model

    model = Pointnet_cls()
    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)


    # Optimizer
    remain_epoch=50

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs+remain_epoch)
    # lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)


    best_source_test_acc = 0
    best_target_test_acc1 = 0
    best_target_test_acc2 = 0

    for epoch in range(max_epoch):

        lr_schedule.step(epoch=epoch)
        print(lr_schedule.get_lr())
        writer.add_scalar('lr', lr_schedule.get_lr(), epoch)

        model.train()
        loss_total = 0
        data_total = 0

        for batch_idx, batch_s in enumerate(source_train_dataloader):

            data, label = batch_s

            data = data.to(device=device)
            label = label.to(device=device).long()

            output_s = model(data)

            loss_s = criterion(output_s, label)

            loss_s.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss_s.item() * data.size(0)
            data_total += data.size(0)

            if (batch_idx + 1) % 10 == 0:
                print('Train:{} [{} /{}  loss: {:.4f} \t]'.format(
                epoch, data_total, num_source_train, loss_total/data_total))


        with torch.no_grad():
            model.eval()

            # ------------Source------------
            loss_total = 0
            correct_total = 0
            data_total = 0
            acc_class = torch.zeros(10, 1)
            acc_to_class = torch.zeros(10, 1)
            acc_to_all_class = torch.zeros(10, 10)

            for batch_idx, (data, label) in enumerate(source_test_dataloader):

                data = data.to(device=device)
                label = label.to(device=device).long()
                output = model(data)
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)

                acc = pred == label

                for j in range(0, 10):
                    label_j_list = (label == j)
                    acc_class[j] += (pred[acc] == j).sum().cpu().float()
                    acc_to_class[j] += label_j_list.sum().cpu().float()
                    for k in range(0, 10):
                        acc_to_all_class[j, k] += (pred[label_j_list] == k).sum().cpu().float()

                loss_total += loss.item() * data.size(0)
                correct_total += torch.sum(pred == label)
                data_total += data.size(0)

            pred_loss = loss_total / data_total
            pred_acc = correct_total.double() / data_total

            if pred_acc > best_source_test_acc:
                best_source_test_acc = pred_acc
            for j in range(0, 10):
                for k in range(0, 10):
                    acc_to_all_class[j, k] = acc_to_all_class[j, k] / acc_to_class[j]
            print('Source Test:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Source Test Acc: {:.4f}]'.format(
                epoch, pred_acc, pred_loss, best_source_test_acc
            ))
            writer.add_scalar('accs/source_test_acc', pred_acc, epoch)

            # ------------Target1------------
            loss_total = 0
            correct_total = 0
            data_total = 0
            acc_class = torch.zeros(10,1)
            acc_to_class = torch.zeros(10,1)
            acc_to_all_class = torch.zeros(10,10)

            for batch_idx, (data,label) in enumerate(target_test_dataloader1):

                data = data.to(device=device)
                label = label.to(device=device).long()
                output = model(data)
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)

                acc = pred == label

                for j in range(0, 10):
                    label_j_list = (label == j)
                    acc_class[j] += (pred[acc] == j).sum().cpu().float()
                    acc_to_class[j] += label_j_list.sum().cpu().float()
                    for k in range(0, 10):
                        acc_to_all_class[j, k] += (pred[label_j_list] == k).sum().cpu().float()

                loss_total += loss.item() * data.size(0)
                correct_total += torch.sum(pred == label)
                data_total += data.size(0)

            pred_loss = loss_total/data_total
            pred_acc = correct_total.double()/data_total

            if pred_acc > best_target_test_acc1:
                best_target_test_acc1 = pred_acc
            for j in range(0, 10):
                for k in range(0, 10):
                    acc_to_all_class[j, k] = acc_to_all_class[j, k]/acc_to_class[j]
            print ('Target 1:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Target 1 Acc: {:.4f}]'.format(
            epoch, pred_acc, pred_loss, best_target_test_acc1
            ))
            writer.add_scalar('accs/target1_test_acc', pred_acc, epoch)


            # ------------Target2------------
            loss_total = 0
            correct_total = 0
            data_total = 0
            acc_class = torch.zeros(10, 1)
            acc_to_class = torch.zeros(10, 1)
            acc_to_all_class = torch.zeros(10, 10)

            for batch_idx, (data, label) in enumerate(target_test_dataloader2):

                data = data.to(device=device)
                label = label.to(device=device).long()
                output = model(data)
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)

                acc = pred == label

                for j in range(0, 10):
                    label_j_list = (label == j)
                    acc_class[j] += (pred[acc] == j).sum().cpu().float()
                    acc_to_class[j] += label_j_list.sum().cpu().float()
                    for k in range(0, 10):
                        acc_to_all_class[j, k] += (pred[label_j_list] == k).sum().cpu().float()

                loss_total += loss.item() * data.size(0)
                correct_total += torch.sum(pred == label)
                data_total += data.size(0)

            pred_loss = loss_total / data_total
            pred_acc = correct_total.double() / data_total

            if pred_acc > best_target_test_acc2:
                best_target_test_acc2 = pred_acc
            for j in range(0, 10):
                for k in range(0, 10):
                    acc_to_all_class[j, k] = acc_to_all_class[j, k] / acc_to_class[j]
            print('Target 2:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Target 2 Acc: {:.4f}]'.format(
                epoch, pred_acc, pred_loss, best_target_test_acc2
            ))
            writer.add_scalar('accs/target2_test_acc', pred_acc, epoch)



if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))

