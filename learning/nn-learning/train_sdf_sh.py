#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import yaml

import os
import matplotlib.pyplot as plt
from sdf.robot_sdf import RobotSdfCollisionNet

import tqdm
b=50000
def create_dataset(robot_name):
    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}
    #    data = loadmat('../weights/table_beeger.mat')['data']
    data = loadmat('../data-sampling/datasets/data_mesh.mat')['dataset']
    # idx_good = (data[:,10:-1] > -0.03).all(1)
    # data = data[idx_good, :]
    # data = data[torch.randperm(data.shape[0])]
    L1 = 0
    L2 = int(0.95 * data.shape[0])
    print(L1, L2)
    n_size = L2
    train_ratio = 0.98
    test_ratio = 0.01
    val_ratio = 1 - train_ratio - test_ratio


    idx_train = np.arange(0, int(n_size * train_ratio))
    idx_val = np.arange(idx_train[-1] + 1, int(n_size * (train_ratio + test_ratio)))
    idx_test = np.arange(idx_val[-1] + 1, int(n_size))

    x = torch.Tensor(data[L1:L2, 0:10]).to(device, dtype=torch.float16)
    y = 100 * torch.Tensor(data[L1:L2, 10:]).to(device, dtype=torch.float16)
    #y[y<0]*=5
    #y[y==0] = 1
    dof = x.shape[1]
    s = 256
    n_layers = 5
    b = 350000
    p = 0.0 #!! How much you make zero of node, not activate. (ex. 1.0 means that you make all node zero)
    skips = []
    fname = 'model/sdf_%dx%d_mesh_%d_drop_%1f.pt'%(s,n_layers,b,p)
    if skips == []:
        n_layers-=1
    nn_model = RobotSdfCollisionNet(in_channels=dof, out_channels=y.shape[1], layers=[s] * n_layers, skips=skips, dropout_ratio=p)
    #nn_model.load_weights('../scripts/sdf_convex_256x5_mesh.pt', tensor_args)
    #nn_model.load_weights('../scripts/gridsearch/5_sdf_convex_512x5.pt', tensor_args)
    #nn_model.load_weights('../scripts/raesdf_256x5_mesh.pt', tensor_args)

    nn_model.model.to(**tensor_args)

    model = nn_model.model
    nelem = sum([param.nelement() for param in model.parameters()])
    print(repr(model))
    print("Sum of parameters:%d" % nelem)
    #time.sleep(2)
    # load training set:
    x_train = x[idx_train, :]
    y_train = y[idx_train, :]
    y_train_labels = y[idx_train, :]
    y_train_labels[y_train_labels<=0] = -1
    y_train_labels[y_train_labels>0] = 1

    y_val_labels = y[idx_val, :]
    y_val_labels[y_val_labels<=0] = -1
    y_val_labels[y_val_labels>0] = 1

    dataset = torch.utils.data.TensorDataset(x_train, y_train, y_train_labels)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b)
    # x_train_loader = torch.utils.data.DataLoader(x_train, batch_size=1024, shuffle=True)
    # y_train_loader = torch.utils.data.DataLoader(y_train, batch_size=1024, shuffle=True)

    # scale dataset: (disabled because of nerf features!)
    mean_x = torch.mean(x, dim=0) * 0.0
    std_x = torch.std(x, dim=0) * 0.0 + 1.0
    mean_y = torch.mean(y, dim=0) * 0.0
    std_y = torch.std(y, dim=0) * 0.0 + 1.0

    x_val = x[idx_val, :]
    y_val = y[idx_val, :]
    x_test = x[idx_test, :]
    y_test = y[idx_test, :]

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                           threshold=0.01, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-04, verbose=True)
    # print(model)
    # epochs = 100000
    epochs = 2000
    min_loss = 2000.0
    # training:
    e_notsaved = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    idx_close_train = (y_train[:, -1] < 10)
    idx_close_val = (y_val[:, -1] < 10)
    
    
    
    

    for e in range(epochs):
        t0 = time.time()
        loader_tqdm = tqdm.tqdm(dataloader)
        for x ,y, _ in loader_tqdm:
            model.train()
            loss = []
            with torch.cuda.amp.autocast():
                y_pred_train = model.forward(x)
                train_loss = F.mse_loss(y_pred_train, y, reduction='mean')

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss.append(train_loss.item())

        model.eval()
        with torch.cuda.amp.autocast():
            y_pred = model.forward(x_val)
            val_loss = F.mse_loss(y_pred, y_val, reduction='mean')
            ee_loss_close = F.l1_loss(y_pred[idx_close_val][:,-1], y_val[idx_close_val][:,-1], reduction='mean')
        if e == 0:
            min_loss = val_loss
        scheduler.step(val_loss)
        train_loss = np.mean(loss)
        if (val_loss < min_loss):
            e_notsaved = 0
            print('saving model', val_loss.item())
            torch.save(
                {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'norm': {'x': {'mean': mean_x, 'std': std_x},
                            'y': {'mean': mean_y, 'std': std_y}}
                },
                fname)
                #'sdf_convex_256_mlp_nerf_skip.pt')
            min_loss = val_loss
        print(y_pred[0, :])
        print(y_val[0, :])
            # if e > 1500:
            #     break
        print(
            "Epoch: %d (Saved at %d), Train Loss: %4.3f, Validation Loss: %4.3f (%4.3f), Epoch time: %4.3f s, LR = %4.8f" % (
                e, e-e_notsaved, train_loss.item(), val_loss.item(), ee_loss_close.item(), time.time() - t0,
                optimizer.param_groups[0]["lr"]))
        f=open("log/log(b=%d)_dropout_%1f.txt" % (b, p), 'a')
        f.write("Epoch: %d (Saved at %d), Train Loss: %4.3f, Validation Loss: %4.3f (%4.3f), Epoch time: %4.3f s, LR = %4.8f\n" % (
                e, e-e_notsaved, train_loss.item(), val_loss.item(), ee_loss_close.item(), time.time() - t0,
                optimizer.param_groups[0]["lr"]))
        f.close()
        e_notsaved += 1
        #print(abs(y_pred-y_val).mean(0))

    # with torch.no_grad():
    #     x = x_test  # [y_test[:,0] > 0.0]
    #     y = y_test  # [y_test[:,0] > 0.0]
    #
    #     # print(x.shape)
    #     y_pred = model.forward(x)
    #     y_pred = torch.mul(y_pred, std_y) + mean_y
    #     y_test = torch.mul(y, std_y) + mean_y
    #     print(y_test[y_test > 0.0])
    #     print(y_pred[y_test > 0.0])
    #     # print(y_pred.shape, y_test.shape)
    #     loss = F.l1_loss(y_pred, y_test, reduction='mean')
    #     print(torch.median(y_pred), torch.mean(y_pred))
    #     print(loss.item())


if __name__ == '__main__':
    create_dataset('franka')