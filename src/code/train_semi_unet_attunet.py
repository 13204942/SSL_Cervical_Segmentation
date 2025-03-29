import argparse
import logging
import os
import random
import shutil
import sys
import time
import cv2

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from config import get_config
from dataloaders.dataset import (BaseDataSets, RandomGenerator, BaseDataSets_Synapse, TwoStreamBatchSampler)
from dataloaders.dataset_fetal import BaseDataSets_Fetal
from networks.unet import UNet as Unet2D
from networks.att_unet import AttU_Net as AttUnet
from utils import losses, ramps
from utils.dice import DSC
from utils.metrics import dice_score
from utils.losses import info_nce_loss
import albumentations as A


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Semi_Mamba_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    # '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )

parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)
# print(config)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model1 = Unet2D(in_chns=3, class_num=args.num_classes).cuda()
    model2 = AttUnet(img_ch=3, output_ch=args.num_classes).cuda()
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    tr_transforms = A.Compose(
        [
            # A.Resize(height=args.patch_size[0], width=args.patch_size[1]),
            A.Rotate(limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),contrast_limit=(-0.5, 0.5)),
            A.Blur(blur_limit=(3, 3), p=0.3),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.9),
            # A.Normalize(
            #     mean=[0.0, 0.0, 0.0],
            #     std=[1.0, 1.0, 1.0],
            #     max_pixel_value=255.0,
            # ),
         ],
    )   
    
    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.patch_size[0], args.patch_size[1])),
    ])    
    
    
    db_train_labled = BaseDataSets_Fetal(base_dir=args.root_path, split="train", num=50, islabeled=True, 
                                         transform=tr_transforms, default_transform=tensor_transforms) 
    
    db_train_labled_tran = BaseDataSets_Fetal(base_dir=args.root_path, split="train", num=50, islabeled=True, 
                                              transform=tr_transforms, default_transform=tensor_transforms)
    
    db_train_unlabled = BaseDataSets_Fetal(base_dir=args.root_path, split="train", islabeled=False, 
                                           transform=tr_transforms, default_transform=tensor_transforms)
    
    db_train = ConcatDataset([db_train_labled, db_train_labled_tran, db_train_unlabled])
    print(f'total lenght of train data: {len(db_train)}')
    
    db_val = BaseDataSets_Fetal(base_dir=args.root_path, split="val", transform=None, default_transform=tensor_transforms)
    
    total_slices = len(db_train)
    labeled_slice = args.labeled_num
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    
    # print("labeled_idxs: {}, unlabeled_idxs: {}".format(labeled_idxs, unlabeled_idxs))

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)     
    
    model1.train()
    model2.train()
    
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    time1 = int(time.time())
    writer = SummaryWriter(snapshot_path + f'/log_{time1}')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    evalue = DSC()  # metric to find best model
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            outputs1 = model1(volume_batch)
            outputs1_emb = model1.encoder(volume_batch)[-1]
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            
            outputs2, outputs2_emb = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long()) + 
                           dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs]))
            
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long()) + 
                           dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs]))
            
            pseudo_outputs1 = outputs_soft1[args.labeled_bs:].detach()
            pseudo_outputs2 = outputs_soft2[args.labeled_bs:].detach()

            pseudo_supervision1 = dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2)
            pseudo_supervision2 = dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1)
            
            outputs1_hidden = torch.flatten(outputs1_emb, start_dim=1)
            outputs2_hidden = torch.flatten(outputs2_emb, start_dim=1)
            
            info_nce_cnn = []
            info_nce_vit = []

            for i in range(args.labeled_bs):
                for j in range(batch_size-args.labeled_bs):
                    
                    info_nce_1 = info_nce_loss(outputs1_hidden[None,i], outputs1_hidden[None,j+args.labeled_bs])
                    info_nce_2 = info_nce_loss(outputs2_hidden[None,i], outputs2_hidden[None,j+args.labeled_bs])                    

                    info_nce_cnn.append(info_nce_1.item())
                    info_nce_vit.append(info_nce_2.item())
                    
            in_loss1 = np.mean(info_nce_cnn)
            in_loss2 = np.mean(info_nce_vit)
            
            model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*in_loss1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*in_loss2 
            
            loss = model1_loss + model2_loss 

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            
            if iter_num > 0 and iter_num % 250 == 0:
                model1.eval()
                model2.eval()
                
                metric_list1 = 0.0
                metric_list2 = 0.0
                
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_valbatch, label_valbatch = sampled_batch['image'], sampled_batch['label']
                    volume_valbatch, label_valbatch = volume_valbatch.cuda(), label_valbatch.cuda()
                    label_valbatch = torch.squeeze(label_valbatch,1)

                    preds_1 = model1(volume_valbatch)
                    preds_2,_ = model2(volume_valbatch)                    
                    
                    preds_1_soft = torch.softmax(preds_1, dim=1)
                    preds_2_soft = torch.softmax(preds_2, dim=1)
                    
                    dsc1_i = dice_score(preds_1_soft, label_valbatch)
                    dsc2_i = dice_score(preds_2_soft, label_valbatch)                      

                    metric_list1 += dsc1_i
                    metric_list2 += dsc2_i

                performance1 = metric_list1 / len(db_val)
                performance2 = metric_list2 / len(db_val)
                
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                
                performance1 = performance1.item()
                performance2 = performance2.item()                  

                if performance1 > best_performance1 and iter_num > 3000:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)
                    
                if performance2 > best_performance2 and iter_num > 3000:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f ' % (iter_num, performance1))
                logging.info(
                    'iteration %d : model2_mean_dice : %f ' % (iter_num, performance2))                                 
                
                model1.train()
                model2.train()   
            
            if iter_num % 500 == 0:
                image = volume_batch[0, ...] * 255
                image_save = image.type(torch.int64)
                print('-'*20)
                print(f'image shape: {image.shape}')
                cv2.imwrite(f"/root/autodl-tmp/assets_unet_attunet/image_{iter_num}.png",np.transpose(image_save.data.cpu().numpy(),(1,2,0))) 
                writer.add_image('train/Image', image_save, iter_num)

                outputs = torch.argmax(torch.softmax(outputs1, dim=1), dim=1, keepdim=True)
                img1_outputs = np.squeeze(outputs[0]) * 100
                test_output = torch.softmax(outputs1, dim=1)
                img_tensor = test_output[0, ...] * 255
                print('-'*20)
                print(f'outputs1 shape: {outputs1.shape}')
                print(f'outputs shape: {outputs[0].shape}')
                print(f'test_output shape: {test_output.shape}')
                print(f'img_tensor shape: {img_tensor.shape}')
                cv2.imwrite(f"/root/autodl-tmp/assets_unet_attunet/outputs1_{iter_num}.png",img1_outputs.data.cpu().numpy())
                writer.add_image('train/model1_Prediction', outputs[0, ...] * 100, iter_num, dataformats='CHW')
                
                outputs = torch.argmax(torch.softmax(outputs2, dim=1), dim=1, keepdim=True)
                img2_outputs = np.squeeze(outputs[0]) * 100
                test_output = torch.softmax(outputs2, dim=1)
                img_tensor = test_output[0, ...] * 255
                print('-'*20)
                print(f'outputs2 shape: {outputs2.shape}')
                print(f'outputs shape: {outputs[0].shape}')
                print(f'test_output shape: {test_output.shape}')
                print(f'img_tensor shape: {img_tensor.shape}')
                cv2.imwrite(f"/root/autodl-tmp/assets_unet_attunet/outputs2_{iter_num}.png",img2_outputs.data.cpu().numpy())
                writer.add_image('train/model2_Prediction', outputs[0, ...] * 100, iter_num, dataformats='CHW')
                
                labs = label_batch[0, ...]
                labs_tensor = labs * 100
                print('-'*20)
                print(f'label_batch shape: {label_batch.shape}')
                print(f'labs shape: {labs.shape}')
                cv2.imwrite(f"/root/autodl-tmp/assets_unet_attunet/labs_{iter_num}.png",np.transpose(labs_tensor.data.cpu().numpy(),(1,2,0)))
                writer.add_image('train/GroundTruth', labs_tensor, iter_num, dataformats='CHW')   

            if iter_num == max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))
                
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()                
    

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)    