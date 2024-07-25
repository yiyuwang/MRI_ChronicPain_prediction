import os
import glob
import json
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import timm
import fire
import wandb
import inspect
from sklearn.metrics import r2_score


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/HEAL_resnet_experiment_1')
root_path = "/home/yiyuw/projects/MRI_TransferLearning"
scripts_path = f"{root_path}/src"
print("scripts from:", scripts_path)
import sys
sys.path.append(scripts_path)

from dataset import MRIDataset, HEALMRIDataset, ZScoreNormalize, EvenlySpreadSlices, RandomSlicesTransform, NormalizeTo1, RepeatChannelTransform, get_datasets, MultiMRIDataset
import models 
from utils import get_optimizer, compute_accuracy, get_lr_lambda, smoothed_labels_to_age, compute_auc, get_model_dir, extract_epoch
from distributed import init_distributed_device, world_info_from_env



def reload_model(model, optimizer, scheduler, checkpoint_path, device, base_model_name='swin'):
    # reconstruct the model after training:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optim'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def count_params(model):
    """
    Print the total number of parameters in the model.

    Parameters:
    - model: PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def calculate_norm_stats(dataset):
    import numpy as np
    batch_list = []
    for i in range(len(dataset)):
        mri_data, _, _ = dataset.__getitem__(i)
        batch_list.append(mri_data.flatten())


    batch_list = np.array(batch_list)
    mean = batch_list.mean()
    std = batch_list.std()
    max_val = batch_list.max()
    min_val = batch_list.min()

    return mean, std, max_val, min_val



def validate(model, model_name, val_loader, device, loss_fn, num_epochs, epoch, rank=0, label_smoothing=0, regression=False):
    # Validation loop.
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    y_preds = []
    y_trues = []
 
    if not regression: 
        # classification
        total_correct = 0
        total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating epoch {epoch + 1}/{num_epochs}"):
            img = batch[0].to(device)
            y_true = batch[1].to(device).long()
            if regression:
                y_true = y_true.float()
             
            if model_name == 'swin_slice':
                img = img.view(-1, 224, 224).unsqueeze(1)
                img = img.repeat(1, 3, 1, 1)
            else: 
                img = img.unsqueeze(1) # for 1 channel 3d cnn
            y_pred = model(img)
            
            # y_pred_prob = F.sigmoid(y_pred) # for classification
            # print(y_pred_prob)
            val_loss = loss_fn(y_pred, y_true)
            total_val_loss += val_loss.item()
            # auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
            
            #  check if classification or regression:
            y_preds.append(y_pred.cpu())
            y_trues.append(y_true.cpu())
            if not regression: 
                correct_preds = compute_accuracy(y_pred, y_true)
                total_correct += correct_preds
                total_samples += y_true.size(0)
      
        avg_val_loss = total_val_loss / len(val_loader) 

        if regression:
            y_preds = torch.cat(y_preds)
            y_trues = torch.cat(y_trues)
            if label_smoothing:
                # convert smoothed y_pred to regression target (e.g. age)
                y_preds = smoothed_labels_to_age(y_preds)
                y_trues = y_trues.argmax(dim=-1) + 20 # min age
                ##print(f'{y_preds=}')
                ##print(f'{y_trues=}')
            r2 = r2_score(y_trues.numpy(), y_preds.numpy())

            print(f'val_loss: {avg_val_loss}')
            print(f'r2: {r2}')
            wandb.log({'val_loss': avg_val_loss})
            wandb.log({'r2': r2})
        else:
            auc = compute_auc(y_preds, y_trues)
            val_accuracy = total_correct / total_samples
        
            print(f'val_loss: {avg_val_loss}')
            print(f'val_accuracy: {val_accuracy}')
            print(f'auc: {auc}')
            wandb.log({'val_loss': avg_val_loss})
            wandb.log({'val_accuracy': val_accuracy})
            wandb.log({'val_auc': auc})


    # # results:
    # if regression:
    #     res = {'val_loss': avg_val_loss, 'r2': r2}
    # else:
    #     res = {'val_loss': avg_val_loss, 'val_accuracy': val_accuracy, 'val_auc': auc}, 
    
    # return res


def main(batch_size=4, num_epochs = 5, learning_rate=3e-04, model_name='swin_slice', norm_clip=1.0, 
           weight_decay=0, num_slices=10, slice_agg='attn', accumulation_steps=1, pool='mean', 
           freeze_backbone=0, optimizer='adamw', use_lr_scheduler=0, n_warmup_epochs=5, lr_warmup_factor=10,
            base_model_name='swin', label='sex', label_smoothing=0, dropout=0, hidden_dim=32):
    root_path = "/home/yiyuw/projects/MRI_TransferLearning"
    config_args = locals()
    print(f'config_args: {config_args}')
    run = wandb.init(config=config_args,
        project="MRITransferLearning_ukb_heal",
        entity='yiyuwang1')
    run_name = run.name
    print(f'run_name: {run_name}')
    model_dir = get_model_dir(run_name, root_path)
    print(f'model_dir: {model_dir}')
    data_df = pd.read_csv(f'{root_path}/data/MRI_TransferLearning_labels.csv')
    # device = rank #device = torch.device(f"cuda:{rank}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel = False

    transforms = [
        ZScoreNormalize(),
        T.Resize((224,224)), 
    ]
    
    print('Initializing model..')
    if model_name == 'swin_slice':
        if base_model_name == 'swin':
            base_model = timm.create_model('swin_large_patch4_window7_224.ms_in22k', pretrained=True)
        elif base_model_name == 'resnet':
            base_model = timm.create_model('resnet50', pretrained=True)
        else:
            raise ValueError(f'{base_model_name} not among valid choices.')
        
        transforms = [
            T.Resize((224,224)),
            NormalizeTo1(), 
            #RepeatChannelTransform() 
        ]
        transforms.append(T.Normalize(mean=[0.4850], std=[0.2290])) # normalization
        transforms.append(EvenlySpreadSlices(desired_slices=num_slices))
 
        # Cut off head from pre-trained model:
        base_model = nn.Sequential(*list(base_model.children())[:-1]) 
        num_features = 1536 if base_model_name == 'swin' else 2048 # resnet
        num_labels = 2 if label in ['sex','group'] else 1
        if label_smoothing:
            num_labels = 71 # hard-coded for now, for age prediction

        model = models.SliceAttentionModel(
                base_model=base_model, 
                num_slices=num_slices, 
                num_features=num_features, num_labels=num_labels,
                hidden_dim=hidden_dim,
                slice_agg=slice_agg, 
                pool=pool,
                base_model_name=base_model_name,
                dropout=dropout
        )
        # freeze backbone:
        if freeze_backbone:
            model.freeze_backbone()

    elif model_name == '3dcnn':
        model = models.Simple3DCNN(num_classes=2)
    elif model_name == 'res3dcnn':
        model = models.Residual3DCNN(num_classes=2)
    else:
        raise ValueError(f'{model_name} is not among valid options.')

    count_params(model)
    model = model.to(device)
    if optimizer == 'adamw':
        optimizer = get_optimizer(model, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=0, centered=True)
    
    
    if use_lr_scheduler:
        from torch.optim.lr_scheduler import LambdaLR
        lr_lambda = get_lr_lambda(num_epochs, n_warmup_epochs, lr_warmup_factor)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


    # cload if there are prior checkpoints:
    prior_epoch = 0
    checkpoint_path = glob.glob(f'{model_dir}/train-UKB_{model_name}_epoch-*.pt')
    checkpoint_path = sorted(checkpoint_path, key=extract_epoch)

    print(checkpoint_path)
    if len(checkpoint_path)>1:
        max_epoch_path = checkpoint_path[-1]
        # model, optimizer, scheduler, prior_epoch = reload_model(model, optimizer, scheduler, checkpoint_path, device, base_model_name=base_model_name)
        checkpoint = torch.load(max_epoch_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optim'])
        if use_lr_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        prior_epoch = checkpoint['epoch']
        
        print(f'Loaded model from {checkpoint_path} at epoch {prior_epoch}')        
        
    count_params(model)
    model = model.to(device)

    # calculate mean and std of the data:
    # init_transforms = T.Compose([T.Resize((224,224)), EvenlySpreadSlices(desired_slices=num_slices)])
    # temp_dataset = MultiMRIDataset(root_path, transform=init_transforms, data_df= data_df, label=label, label_smoothing=0) 
    # data_mean, data_std, max_val, min_val = calculate_norm_stats(temp_dataset)
    # print(f'{data_mean=}, {data_std=}, {max_val=}, {min_val=}')

    
    
    composed_transforms = T.Compose(transforms)
    dataset = MultiMRIDataset(root_path, transform=composed_transforms, data_df= data_df, label=label, label_smoothing=label_smoothing)

    # Load the splits from the json file
    with open(f'{root_path}/data/MRI_TransferLearning_split_3way.json', 'r') as f:
        split_dict = json.load(f)


	# Get data splits:
    print('Loading datasets..')
    train_dataset, val_dataset, test_dataset = get_datasets(dataset, split_dict)

    # checking labels and patient_ids are correct:
    for i in range(3):
        random_id = torch.randint(0, len(train_dataset), (1,)).item()
        _, y, pat_id = train_dataset.__getitem__(random_id)
        print(f"subject {pat_id} has label {y}")

        random_id = torch.randint(0, len(val_dataset), (1,)).item()
        _, y, pat_id = val_dataset.__getitem__(random_id)
        print(f"subject {pat_id} has label {y}")

        random_id = torch.randint(0, len(test_dataset), (1,)).item()
        _, y, pat_id = test_dataset.__getitem__(random_id)
        print(f"subject {pat_id} has label {y}")


    # Initialize distributed device:
    rank = 0

    # train_sampler = DistributedSampler(train_dataset)
    # val_sampler = DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            # shuffle=False, # for DDP 
            shuffle=True,
            drop_last=True, 
            num_workers=4,
            # sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            # shuffle=False, 
            shuffle=True,
            drop_last=True, 
            num_workers=4,
            # sampler=val_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=True, 
            num_workers=4,
    )
    
    # DP Version:
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    #val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True) # due to multigpu

        
    # Training parameters
    if label in ['sex','group']:
        loss_fn = nn.CrossEntropyLoss()
    else:
        if label_smoothing:
            # age classification with smoothed labels 
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()

    # Determine if regression labels:
    regression = False
    if 'MSE' in loss_fn.__repr__() or label_smoothing:
        regression = True

    # initial val step:
    validate(model, model_name, test_loader, device, loss_fn,  num_epochs, epoch=-1, rank=rank, 
                label_smoothing=label_smoothing, regression=regression)
    
    optimizer.zero_grad() 
    step = 0

    print(f'Starting training from epoch {prior_epoch} to {num_epochs}')
    
    for epoch in range(num_epochs - prior_epoch):

        # train_sampler.set_epoch(epoch)
        model.train()  # Set the model to training mode

        # Log learning rate at the start of the epoch
        lr = optimizer.param_groups[0]['lr']


        total_train_loss = 0

        wandb.log({'Epoch': epoch})
        wandb.log({"learning_rate": lr})

        # Training step 
        for batch in tqdm(train_loader):
            img = batch[0].to(device)  # Move data to GPU
            #print(f'{img.shape=}')
            y_true = batch[1].to(device).long()  # Move labels to GPU
            if regression:
                y_true = y_true.float()
 
            if model_name == 'swin_slice':
                # collapsing depth and batch dimension
                
                img = img.view(-1, 224, 224).unsqueeze(1)
                #img = img.view(-1, 3, 224, 224)
                #print(f'Collapsed img shape: {img.shape}')
                # repeat channel dim (3 expected)
                img = img.repeat(1, 3, 1, 1)
            else: 
                img = img.unsqueeze(1) # for 1 channel 3d cnn

            y_pred = model(img)

            loss = loss_fn(y_pred, y_true)
            total_train_loss += loss.item() 

            loss.backward()  # Compute gradients

            # if rank.index == 0:
                # Log gradient norm for each layer using wandb
            gradient_norms = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    gradient_norms[f"gradient_norm/{name}"] = grad_norm
            wandb.log(gradient_norms) 
        
            # Update weights every 'accumulation_steps' iterations
            if (step +1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=norm_clip)
                optimizer.step()
                optimizer.zero_grad()
            
            print(f'train_loss: {loss.item()}')
            wandb.log({'train_loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader) 
        
        wandb.log({'avg_train_loss': avg_train_loss})
        print(f'avg_train_loss: {avg_train_loss}')

        # Val step
        validate(model, model_name, val_loader, device, loss_fn, num_epochs, epoch, 
                rank, label_smoothing, regression=regression)

        # Update LR:
        if use_lr_scheduler:
            scheduler.step()

        # save model:
        if epoch + 1 % 10 == 0:
            if use_lr_scheduler:
                torch.save({
                    'net': model.state_dict(),
                    'optim': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch}, f'{model_dir}/train-UKB_{model_name}_label-{label}_epoch-{epoch}.pt')
            else: 
                torch.save({
                    'net': model.state_dict(),
                    'optim': optimizer.state_dict(), 
                    'epoch': epoch}, f'{model_dir}train-UKB_{model_name}_label-{label}_epoch-{epoch}.pt')
    # validate(model, model_name, test_loader, device, loss_fn, num_epochs+1, epoch+1, rank, label_smoothing, regression=regression)

if __name__ == "__main__":
   #main(label='group',base_model_name='swin', num_epochs = 50)
    
    main(
        accumulation_steps=1,
        base_model_name="resnet",
        batch_size=16,
        dropout=0.7,
        freeze_backbone=1,
        hidden_dim=128,
        label="sex",
        label_smoothing=0,
        learning_rate=0.0001,
        lr_warmup_factor=10,
        model_name="swin_slice",
        n_warmup_epochs=5,
        norm_clip=1,
        num_epochs=200,
        num_slices=64,
        optimizer="adamw",
        pool="max",
        slice_agg="attn",
        use_lr_scheduler=1,
        weight_decay=0.001
    )