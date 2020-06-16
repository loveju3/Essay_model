# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:10:58 2020

@author: rong
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from torchmeta.transforms import Categorical, ClassSplitter, Rotation

from torchmeta.datasets import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter

from model import ConvolutionalNeuralNetwork, store_grad, project2cone2, overwrite_grad
from utils import update_parameters, get_accuracy

def do_past_batch(model, batch_idx):
    for tt in range(len(model.observed_tasks) - 1):
        past_loss = torch.tensor(0., device=args.device)
        model.zero_grad() 
        past_task = model.observed_tasks[tt]
        # to get the first fast weights
        p_train_inputs = model.memory_data_spt[past_task].cuda()
        p_train_targets = model.memory_labs_spt[past_task].cuda()
        p_test_inputs = model.memory_data_qry[past_task].cuda()
        p_test_targets = model.memory_labs_qry[past_task].cuda()
        
        for task_idx, (train_input, train_target, test_input,
                       test_target) in enumerate(zip(p_train_inputs, p_train_targets,
                        p_test_inputs, p_test_targets)):
            
            train_logit = model(train_input)
            inner_loss = F.cross_entropy(train_logit, train_target)

            model.zero_grad()
            params = update_parameters(model, inner_loss,
                    step_size=args.step_size, first_order=args.first_order)
            model.zero_grad()
            test_logit = model(test_input, params=params)
            past_loss += F.cross_entropy(test_logit, test_target)
        
        past_loss.div_(args.batch_size)
        past_loss.backward()
        store_grad(model.parameters, model.grads, model.grad_dims,
                   batch_idx)

def train(args, model, meta_optimizer):
    
    dataset = MiniImagenet(args.folder, 
                           num_classes_per_task=args.num_ways, 
                           meta_train=True,
                           meta_val=False, meta_test=False, meta_split=None,
                           transform=Compose([Resize(84), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), target_transform=Categorical(num_classes=5), 
                           dataset_transform=None,class_augmentations=[Rotation([90, 180, 270])], download=True)
    
    print('args.batch_size: %d',args.batch_size)
    
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=args.num_shots, num_test_per_class=15)
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    model.train()

    with tqdm(dataloader, total=args.num_batches) as pbar:
        accs = []
        
        for batch_idx, batch in enumerate(pbar):
            print(batch_idx)
            cnt = batch_idx
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)
            
            if len(model.observed_tasks) < 2:
               model.observed_tasks.append(batch_idx)
               model.memory_data_spt[batch_idx].copy_(train_inputs)
               model.memory_labs_spt[batch_idx].copy_(train_targets)
               model.memory_data_qry[batch_idx].copy_(test_inputs)
               model.memory_labs_qry[batch_idx].copy_(test_targets)
            else:
                
                batch_idx = -1
                
                for i in range(len(model.memory_data_spt) - 1):
                    model.memory_data_spt[i] = model.memory_data_spt[i+1]
                    model.memory_labs_spt[i] = model.memory_labs_spt[i+1]
                    model.memory_data_qry[i] = model.memory_data_qry[i+1]
                    model.memory_labs_qry[i] = model.memory_labs_qry[i+1]
                   
                model.memory_data_spt[-1].copy_(train_inputs)
                model.memory_labs_spt[-1].copy_(train_targets)
                model.memory_data_qry[-1].copy_(test_inputs)
                model.memory_labs_qry[-1].copy_(test_targets)
                
            #
            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            
            # compute loss of past batch and store their gradient on each tasks in past batch
            # past_task is a batch
            if len(model.observed_tasks) > 1:
                do_past_batch(model, batch_idx)
                    
            model.zero_grad()
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
               
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = update_parameters(model, inner_loss,
                    step_size=args.step_size, first_order=args.first_order)

                
                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)
            accs.append(accuracy)
            outer_loss.backward()
            store_grad(model.parameters, model.grads, model.grad_dims,
                           batch_idx)
            
            if len(model.observed_tasks) > 1:
               indx = torch.LongTensor(model.observed_tasks[:-1])
               dotp = torch.mm(model.grads.index_select(0, indx),
                               model.grads[batch_idx, : ].unsqueeze(1))
               
               if (dotp < 0).sum() != 0:
                   model.overwrite_num += 1
                   project2cone2(model.grads[:, batch_idx].unsqueeze(1),
                                 model.grads.index_select(1, indx), model.margin)
                   overwrite_grad(model.parameters, model.grads[batch_idx, :],
                                      model.grad_dims)
                        
            meta_optimizer.step()
            pbar.set_postfix(train_accuracy='{0:.4f}'.format(accuracy.item()))
            if cnt >= args.num_batches:
                return accs
                break

def validation(args, model):
    dataset = MiniImagenet(args.folder, 
                           num_classes_per_task=args.num_ways, 
                           meta_train=False,
                           meta_val=True, meta_test=False, meta_split=None,
                           transform=Compose([Resize(84), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), target_transform=Categorical(num_classes=5), 
                           dataset_transform=None,class_augmentations=None, download=True)
    
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=args.num_shots, num_test_per_class=15)
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    
    model.eval()

    with tqdm(dataloader, total=args.num_batches) as pbar:
        accs = []
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()
            
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)
            
            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = update_parameters(model, inner_loss,
                    step_size=args.step_size, first_order=args.first_order)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            accuracy.div_(args.batch_size)
            accs.append(accuracy)
            pbar.set_postfix(test_accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_testbatches:
                return accs
                break
           
def test(args, model):
    print('testing')
    dataset = MiniImagenet(args.folder, num_classes_per_task=args.num_ways, meta_train=False,
                 meta_val=False, meta_test=True, meta_split=None,
                 transform=Compose([Resize(84), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), target_transform=Categorical(num_classes=5), 
                 dataset_transform=None, class_augmentations=None, download=False)
    
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=args.num_shots, num_test_per_class=15)
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    
    model.eval()

    with tqdm(dataloader, total=args.num_testbatches) as pbar:
        accs = []
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()
            
            print(batch_idx)
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)
            
            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                for i in range(10):
                    train_logit = model(train_input)
                    inner_loss = F.cross_entropy(train_logit, train_target)

                    model.zero_grad()
                    params = update_parameters(model, inner_loss,
                        step_size=args.step_size, first_order=args.first_order)
                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            accuracy.div_(args.batch_size)
            accs.append(accuracy)
            pbar.set_postfix(test_accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_testbatches:
                return accs
                break

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    
    parser.add_argument('--epochs', type=int, default=600,
        help='Number of examples epoch (k in "k-shot", default: 5).')
    
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    ## '--folder'
    parser.add_argument('--folder', type=str, default='C:\\Users\\rong\\Anaconda3\\test\\torchmeta\\CM_mini_imagenet\\data',
        help='Path to the folder the data is downloaded to.')
   
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.01,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=32,
        help='Number of channels for each convolutional layer (default: 32).')

    parser.add_argument('--output-folder', type=str, default='C:\\Users\\rong\\Anaconda3\\test\\torchmeta\\CM_mini_imagenet',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default= 100,
        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=0,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', type=str, default='True',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', type=str, default='True',
        help='Use CUDA if available.')
    #--------------------test args------------------------------------
    
    parser.add_argument('--num-testbatches', type=int, default=5,
        help='Number of batches the model is tested over (default: 1).')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')
    
    writer = SummaryWriter('runs/CM_ImgNet_5w5s/two_batch_allsetting_fix')

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model_0 = ConvolutionalNeuralNetwork(3, args.num_ways, args,
        hidden_size=args.hidden_size)
    
    model_0.to(device=args.device)
    meta_optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=1e-3)
    
    print(args.device)
    
    for epoch in range(args.epochs):
        
        # model_0
        tr_acc_0 = train(args, model_0, meta_optimizer_0)
        tr_acc_0 = torch.mean(torch.stack(tr_acc_0))
        print('tr_acc_0',tr_acc_0)
        writer.add_scalar('CM_Mini_training_5W5S_acc_0',
                            tr_acc_0,
                            (epoch))
        
        val_acc_0 = validation(args, model_0)
        val_acc_0 = torch.mean(torch.stack(val_acc_0))
        print('val_acc',val_acc_0)
       
        writer.add_scalar('CM_Mini_val_5W5S_acc_0',
                            val_acc_0,
                            (epoch))
        
        test_acc_0 = test(args, model_0)
        test_acc_0 = torch.mean(torch.stack(test_acc_0))
        print('mean_test_acc',test_acc_0)
        print('epoch', epoch)
        writer.add_scalar('MAML_Mini_test_5W5S_acc_0',
                    test_acc_0,
                      (epoch))

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    model_1 = ConvolutionalNeuralNetwork(3, args.num_ways, args,
        hidden_size=args.hidden_size)
    model_1.to(device=args.device)
    meta_optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        
        # model_1
       
        tr_acc_1 = train(args, model_1, meta_optimizer_1)
        tr_acc_1 = torch.mean(torch.stack(tr_acc_1))
        print('tr_acc_1',tr_acc_1)
        writer.add_scalar('CM_Mini_training_5W5S_acc_1',
                            tr_acc_1,
                            (epoch))

        val_acc_1 = validation(args, model_1)
        val_acc_1 = torch.mean(torch.stack(val_acc_1))
        print('val_acc',val_acc_1)
        writer.add_scalar('CM_Mini_val_5W5S_acc_1',
                            val_acc_1,
                            (epoch))
        
        test_acc_1 = test(args, model_1)
        test_acc_1 = torch.mean(torch.stack(test_acc_1))
        print('test_acc', test_acc_1)
        print('epoch', epoch)
        writer.add_scalar('CM_Mini_test_5W5S_acc_1',
                            test_acc_1,
                            (epoch))
    
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)
    model_2 = ConvolutionalNeuralNetwork(3, args.num_ways, args,
        hidden_size=args.hidden_size)
    model_2.to(device=args.device)
    meta_optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        # moedl_3
       
        tr_acc_2 = train(args, model_2, meta_optimizer_2)
        tr_acc_2 = torch.mean(torch.stack(tr_acc_2))
        print('tr_acc_2',tr_acc_2)
        writer.add_scalar('CM_Mini_training_5W5S_acc_2',
                            tr_acc_2,
                            (epoch))
          
        val_acc_2 = validation(args, model_2)
        val_acc_2 = torch.mean(torch.stack(val_acc_2))
        print('val_acc',val_acc_2)
        writer.add_scalar('CM_Mini_val_5W5S_acc_2',
                           val_acc_2,
                            (epoch))

        test_acc_2 = test(args, model_2)
        test_acc_2 = torch.mean(torch.stack(test_acc_2))
        print('test_acc',test_acc_2)
        print('epoch', epoch)
        writer.add_scalar('CM_Mini_test_5W5S_acc_2',
                            test_acc_2,
                            (epoch))
        
        


    PATH_0 = './CM_IMG_net_0_all_fix.pth'
    torch.save( model_0.state_dict(), PATH_0)
    
    PATH_1 = './CM_IMG_net_1_all_fix.pth'
    torch.save( model_1.state_dict(), PATH_1)
    
    PATH_2 = './CM_IMG_net_2_all_fix.pth'
    torch.save( model_2.state_dict(), PATH_2)