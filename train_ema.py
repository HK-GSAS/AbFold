import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.hipify.hipify_python import bcolors
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from abfold.config import config
from abfold.data_module import AbFoldDataset, UnsupervisedDataset
from abfold.model import AbFold
from abfold.utils.lr_schedulers import AlphaFoldLRScheduler
from abfold.utils.unsupervised_loss import AlphaFoldUnsupervisedLoss

device = "cuda:0" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter()


def update_state_dict_(update, state_dict, decay=0.999):
    with torch.no_grad():
        for k, v in update.items():
            stored = state_dict[k]
            if not isinstance(v, torch.Tensor):
                update_state_dict_(v, stored, decay)
            else:
                diff = stored - v
                diff *= 1 - decay
                stored -= diff


def optimizer_params_to_device(optimizer, aim_device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(aim_device)


def print_val_grad(net):
    v_n = []
    v_v = []
    v_g = []
    for name, parameter in net.named_parameters():
        v_n.append(name)
        v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
        v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
    for i in range(len(v_n)):
        if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
            color = bcolors.FAIL + '*'
        else:
            color = bcolors.OKGREEN + ' '
        print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
        print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))


def tensor_dict_to_device(tensor_dict: dict, device):
    new_dict = {}
    for key, val in tensor_dict.items():
        if type(val) == torch.Tensor:
            new_dict[key] = val.to(device, non_blocking=True)
        elif type(val) == dict:
            new_dict[key] = tensor_dict_to_device(val, device)
    return new_dict


def adjust_dim(input):
    keys = ["frames", "sidechain_frames", "unnormalized_angles", "angles", "positions", "t_global_tensor_7"]
    half = input['sm']['frames'].shape[0] // 2
    for key in keys:
        input['sm'][key] = torch.cat((input['sm'][key][:half], input['sm'][key][half:]), dim=1)


def unsqueeze_sm(input):
    keys = ["frames", "sidechain_frames", "unnormalized_angles", "angles", "positions", "t_global_tensor_7"]
    for key in keys:
        input['sm'][key] = input['sm'][key].unsqueeze(dim=0)


def cat_input(X1, X2):
    X = dict()
    for key in X1.keys():
        X[key] = torch.cat([X1[key], X2[key]], dim=0)

    return X


def split_output(output, sizes):
    keys = ["frames", "sidechain_frames", "unnormalized_angles", "angles", "positions", "t_global_tensor_7"]
    output1 = dict()
    output2 = dict()
    for key, val in output.items():
        if isinstance(val, dict):
            output1[key], output2[key] = split_output(val, sizes)
        else:
            if key in keys:
                output1[key], output2[key] = torch.split(val, sizes, dim=1)
            else:
                output1[key], output2[key] = torch.split(val, sizes, dim=0)
    return output1, output2


def train(dataloader1, dataloader2, student, teacher, loss_fn, optimizer, scheduler, epoch):
    size1 = dataloader1.batch_size
    size2 = dataloader2.batch_size
    size = int(len(dataloader2.dataset) / size2)
    total_loss = 0
    for i, (X1, L2) in enumerate(tqdm(zip(dataloader1, dataloader2))):
        X2, y2 = L2
        X = cat_input(X1, X2)
        X, y2 = tensor_dict_to_device(X, device), tensor_dict_to_device(y2, device)

        pred_student = student(X)
        pred_teacher = teacher(X)
        adjust_dim(pred_student)
        adjust_dim(pred_teacher)
        pred_student1, pred_student2 = split_output(pred_student, [size1, size2])
        pred_teacher1, pred_teacher2 = split_output(pred_teacher, [size1, size2])
        loss = loss_fn(pred_student1, pred_teacher1, pred_student2, y2)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # print_val_grad(abfold)
        torch.nn.utils.clip_grad_norm_(student.module.parameters(), 0.1)

        optimizer.step()
        scheduler.step()
        update_state_dict_(student.module.state_dict(), teacher.module.state_dict())
        writer.add_scalar('loss', loss, epoch * size + i)
        total_loss += loss

        if i % 100 == 0:
            loss, current = loss.item(), i
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    writer.add_scalar('epoch_loss_mean', total_loss / size, epoch)


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    global device
    multi_gpu = args.multi_gpu
    if multi_gpu:
        device_ids = [int(i) for i in args.gpus.split(',')]
        device = device_ids[0]
    max_lr = args.max_lr
    batch_size = args.batch_size
    unsupervised_repr_dir = args.unsupervised_repr_dir
    unsupervised_fasta_dir = args.unsupervised_fasta_dir
    label_dir = args.label_dir
    non_struct_batch_size = round(args.non_struct_data_ratio_per_epoch * batch_size)
    non_struct_data = UnsupervisedDataset(unsupervised_repr_dir, unsupervised_fasta_dir,
                                          point_feat_dir=args.non_struct_point_feat_dir)
    non_struct_dataloader = DataLoader(non_struct_data, batch_size=non_struct_batch_size, num_workers=batch_size,
                                       shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True)
    struct_batch_size = batch_size - non_struct_batch_size
    struct_data = AbFoldDataset(label_dir, point_feat_dir=args.point_feat_dir)
    struct_dataloader = DataLoader(struct_data, batch_size=struct_batch_size, num_workers=args.num_workers,
                                   shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True)

    student = AbFold(config)
    teacher = AbFold(config)
    loss_fn = AlphaFoldUnsupervisedLoss(config['loss'], ratio=args.non_struct_loss_ratio)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-4,
        eps=1e-6
    )
    scheduler = AlphaFoldLRScheduler(optimizer, max_lr=max_lr)
    if multi_gpu:
        student = torch.nn.DataParallel(student, device_ids=device_ids)
        teacher = torch.nn.DataParallel(teacher, device_ids=device_ids)

    student, teacher = student.to(device), teacher.to(device)
    loss_fn = loss_fn.to(device)
    optimizer_params_to_device(optimizer, device)
    teacher.module.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    start_epoch = 0
    checkpoint_name = args.checkpoint_name

    if args.resume:
        if os.path.isfile(checkpoint_name):
            checkpoint = torch.load(checkpoint_name)
            start_epoch = checkpoint['epoch'] + 1
            student.module.load_state_dict(checkpoint['student'])
            teacher.module.load_state_dict(checkpoint['teacher'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    time_0 = time.time()
    save_time = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        optimizer.zero_grad()
        optimizer.step()
        print(f"Epoch {epoch}\n-------------------------------")
        train(non_struct_dataloader, struct_dataloader, student, teacher, loss_fn, optimizer, scheduler, epoch)
        checkpoint = {
            'epoch': epoch,
            'student': student.module.state_dict(),
            'teacher': teacher.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        time_1 = time.time()
        torch.save(checkpoint, checkpoint_name)
        save_time += time.time() - time_1
    print(f'epoch time: {time.time() - time_0}s')
    print(f'save time: {save_time}s')
    print("Done!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsupervised_repr_dir", type=str,
                        default="data/abfold/train/unsupervised_repr",
                        help="Directory containing precomputed representations")
    parser.add_argument("--unsupervised_fasta_dir", type=str,
                        default="data/abfold/train/unsupervised_fasta",
                        help="Directory containing fasta files")
    parser.add_argument("--label_dir", type=str, default="data/abfold/train/supervised_label",
                        help="Directory containing label files")
    parser.add_argument("--resume", type=str, default=False,
                        help="If resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=300, help="training epochs")
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_ema")
    parser.add_argument("--gpus", type=str, default='0,1', help="use comma as separator")
    parser.add_argument("--max_lr", type=float, default=0.0001, help="max learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="training gpu")
    parser.add_argument("--non_struct_data_ratio_per_epoch", type=float, default=0.5,
                        help="the ratio of non-structure data per training epoch")
    parser.add_argument("--non_struct_loss_ratio", type=float, default=0.5,
                        help="the loss ratio of non-structure data per training epoch")
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--point_feat_dir", type=str, default="data/abfold/train/embed",
                        help="Directory containing embedding files generated by point_mae")
    parser.add_argument("--non_struct_point_feat_dir", type=str,
                        default="data/abfold/train/unsupervised_embed",
                        help="Directory containing embedding files generated by point_mae for non-structure data")
    args = parser.parse_args()
    main(args)
