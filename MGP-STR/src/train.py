import os
import sys
import time
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy

from utils import Averager, CTCLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from models import Model
from test_final import validation
from utils import get_args
import utils_dist as utils

import torch.nn as nn
from tqdm.auto import tqdm

import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')

    val_opt = copy.deepcopy(opt)
    val_opt.eval = True
    
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=val_opt)
    valid_dataset, _ = hierarchical_dataset(root=opt.valid_data, opt=val_opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
        
    """ model configuration """
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        state_dict = torch.load(opt.saved_model, map_location='cpu')
        new_state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in new_state_dict:
                if new_state_dict[name].shape != param.shape:
                    print(f"Skipping parameter {name} due to shape mismatch")
                    # 기존 가중치 복사 및 새 가중치 초기화
                    if len(new_state_dict[name].shape) == len(param.shape):
                        min_shape = [min(new_dim, old_dim) for new_dim, old_dim in zip(new_state_dict[name].shape, param.shape)]
                        slices = tuple(slice(0, dim) for dim in min_shape)
                        new_state_dict[name][slices] = param[slices]
                        if len(new_state_dict[name].shape) > 1:
                            nn.init.normal_(new_state_dict[name][min_shape[0]:])
                        else:
                            nn.init.zeros_(new_state_dict[name][min_shape[0]:])
                else:
                    new_state_dict[name] = param
        model.load_state_dict(new_state_dict)


    """ setup loss """
    criterion = nn.CTCLoss(zero_infinity=True).to(device)
    # loss averager
    loss_avg = Averager()
    char_loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    if opt.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000000)

    """ final options """
    with open(f'{opt.saved_path}/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass
    
    start_time = time.time()
    best_accuracy = -1
    iteration = start_iter
            
    print("LR",scheduler.get_last_lr()[0])
    
    pbar = tqdm(total=opt.num_iter, disable=not utils.is_main_process())
    pbar.update(iteration)

    while(True):
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)

        ##
        valid_indices = [i for i, label in enumerate(labels) if len(label) <= opt.batch_max_length]
        image_tensors = image_tensors[valid_indices]
        labels = [labels[i] for i in valid_indices]
        ##

        text, length = converter.encode(labels, batch_max_length= opt.batch_max_length)
        batch_size = image.size(0)
        preds = model(image)
        
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.log_softmax(2).permute(1, 0, 2)
        
        cost = criterion(preds, text, preds_size, length)

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
        optimizer.step()

        loss_avg.add(cost)
        

        pbar.set_postfix({"Train Loss": f"{loss_avg.val():0.5f}"})
        # pbar.update(1)
        if utils.is_main_process() and opt.wandb:
            wandb.log({
                "iteration": iteration,
                "train_loss": loss_avg.val(),
                "learning_rate": scheduler.get_last_lr()[0] if scheduler else opt.lr,
                })

        if utils.is_main_process() and ((iteration + 1) % opt.valInterval == 0):
            elapsed_time = time.time() - start_time
            print("LR",scheduler.get_last_lr()[0])
            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, char_preds, confidence_score, labels, infer_time, length_of_data, _, ned = validation(
                        model, criterion, valid_loader, converter, opt)
                    char_accuracy = current_accuracy
                    cur_best = max(char_accuracy, ned)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] LR: {scheduler.get_last_lr()[0]:0.5f}, Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"char_accuracy":17s}: {char_accuracy:0.3f}, {"NED":5s}: {ned:0.3f}'
                
                if cur_best > best_accuracy:
                    best_accuracy = cur_best
                    torch.save(model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/best_accuracy.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # wandb logging for validation
                if opt.wandb:
                    if opt.Transformer in ["char-str"]:
                        wandb.log({
                            "iteration": iteration,
                            "valid_loss": valid_loss,
                            "char_accuracy": char_accuracy,
                            "best_accuracy": best_accuracy,
                            "NED": (1-ned)
                        })

                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], char_preds[:5], confidence_score[:5]):
                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        if utils.is_main_process() and (iteration + 1) % 5e+3 == 0:
            torch.save(
                model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/iter_{iteration+1}.pth')

        iteration += 1
        pbar.update(1)
        
        if scheduler is not None:
            scheduler.step()
        
        if (iteration + 1) == opt.num_iter:
            print('end the training')
            pbar.close()
            sys.exit()

def init_wandb(cfg):
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project="MGP",
        name=cfg.exp_name,
        config=vars(cfg),
        entity = os.environ.get("WANDB_ENTITY"),
        save_code= True,
        # resume=True,
    )

if __name__ == '__main__':

    opt = get_args()
    # if 'pkl' in opt.character:
    #     with open(opt.character, 'rb') as f:
    #         extended_char = pickle.load(f)
    #     ## '金','整','公','簿' 추후 추가
    #     extended_char.extend(['±',' ','△','※','☑','☐','⓪','①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','⑪','⑫','⑬','⑭','⑮','⑯','⑰','⑱','⑲','⑳','@'])
    #     opt.character = ''.join(extended_char)
    import ocr_dict
    opt.character = ocr_dict.all_chars
    
    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    utils.init_distributed_mode(opt)

    print(opt)
    
    """ Seed and GPU setting """
    
    seed = opt.manualSeed + utils.get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    if utils.is_main_process() and opt.wandb:
        init_wandb(opt)
    train(opt)