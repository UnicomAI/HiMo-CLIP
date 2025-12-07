import torch
#from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
#the original concat_all_gather is abandoned because of no gradient backward
from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import random
import sys
sys.path.append("..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import himo as longclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler
# import warnings
# warnings.filterwarnings("ignore")
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        torch.distributed.nn.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count

def drop_part(caption_list): 
    ### 抛弃部分文本，但一定会留下有效信息
    total = len(caption_list)
    if total == 1:
        return caption_list, 1
    
    ### TODO: 设定抛弃方式
    # drop_num = random.choice(range(1, total)) # 必定抛弃
    drop_num = random.choice(range(0, total)) # 允许不抛弃
    # drop_num = random.choice(range(total//2, total))
    
    pop_idxs = set(random.sample(range(len(caption_list)), drop_num))
    new_list = []
    for i in range(len(caption_list)):
        if i in pop_idxs: continue
        new_list.append(caption_list[i])
    return new_list, len(new_list)

def drop_part_2(caption_list):
    ### 抛弃部分文本，允许全抛
    total = len(caption_list)
    drop_num = random.choice(range(1, total+1))
    if drop_num == total:
        return [""], 0
    
    pop_idxs = set(random.sample(range(len(caption_list)), drop_num))
    new_list = []
    for i in range(len(caption_list)):
        if i in pop_idxs: continue
        new_list.append(caption_list[i])
    return new_list, len(new_list)

class CLIP_Clean_Train():
    def __init__(self, rank, local_rank, args):
        self.rank=rank
        self.local_rank = local_rank
        self.base_model = args.base_model
        if "train/weights/ViT-L-14.pt" in self.base_model: # load official clip model
            self.model, _ = longclip.load_from_clip(self.base_model, device='cpu',download_root=args.download_root)
        else:
            self.model, _ = longclip.load(self.base_model, device='cpu')
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        if args.exp_name == "auto":
            self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)

           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()
        
        if rank == 0:
            print(args)
            print(f"pca_ratio: {args.pca_ratio}")


        # define save_root
        jobname = args.jobname
        self.save_dir = os.path.join(args.save_root, jobname)
        if rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.args = args


    #### 支持accumulate gradient
    def train_epoch(self, dataloader, epoch, start_iter=0, train_losses=None):
        print_freq = 20
        num_batches_per_epoch = len(dataloader)

        accum_steps = self.args.accum_steps  # e.g. 4

        # 在每个 epoch 开始时先置零
        self.optimizer.zero_grad()

        for i, (images, texts, _) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):

            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue

            texts = longclip.tokenize(texts, truncate=True).cuda()

            self.scheduler(step)

            with torch.cuda.amp.autocast():    

                loss_il1, loss_il2 = self.model(images, texts, self.rank, pca_ratio=self.args.pca_ratio)
                loss = loss_il1 + loss_il2

                
                loss = loss / accum_steps

            train_losses.update((loss*accum_steps).detach().item())
            self.scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches_per_epoch:
                # optimizer step & zero grad
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()  # 清空累积的梯度
            
                # torch.cuda.empty_cache() # 去掉cuda上不必要的缓存
        
            current_lr = self.optimizer.param_groups[0]['lr']  # 获取当前学习率
            logs = {"EP": f'{epoch}/{self.args.epochs-1}, {i}/{num_batches_per_epoch}',
                    'gs': step, 
                    "loss": round(train_losses.avg,4), 
                    "lr": current_lr,
                    "loss_il1": round(loss_il1.detach().item(),4), 
                    "loss_il2": round(loss_il2.detach().item(),4)
                    }
            
            if step % print_freq == 0 and self.rank == 0:
                print(logs)


    @torch.no_grad()
    def test_epoch(self, dataloader):
        temp_corr_dict = dict()
        rank = torch.distributed.get_rank()

        for id, (images, text) in enumerate(tqdm(dataloader, disable=(rank != 0))):

            images = images.cuda()
            image_features = self.model.module.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text = longclip.tokenize(text, truncate=True).cuda()
            text_feature = self.model.module.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

            i = 0
            correct = 0
            total = 0

            for i in range(text_feature.shape[0]):
                text = text_feature[i]
                sim = text @ image_features.T
                sim = sim.squeeze()
                correct_i = torch.argmax(sim)

                if i==correct_i:
                    correct = correct + 1
                total = total + 1

        return correct/total
    
    def test(self, epoch=0):
        rank = torch.distributed.get_rank()
        if rank == 0:
            self.model.eval()
            testset = share4v_val_dataset()
            testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True)
            with torch.no_grad():    

                acc = self.test_epoch(testloader)
                print("=====================================")
                print(f"test mean of share4v retrieval: {acc}")
                print("=====================================")

            return
    
    def train(self, resume=False, warmup_length=200):
        trainset = share4v_train_dataset()
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=32, pin_memory=True)
        train_losses = AverageMeter(20) # 记录train loss的


        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
        start_epoch = 0
        resume_iter = 0
        
        for epoch in range(start_epoch, self.num_epoch):
            
            self.train_epoch(train_loader, epoch, start_iter=resume_iter, train_losses=train_losses)
            if self.rank == 0:
                name = "himo.pt"
                # now = datetime.now()
                # formatted_date = now.strftime("%m-%d--%H_%M_%S_")
                #torch.distributed.barrier()
                # torch.save(self.model.module.state_dict(), './checkpoints/'+str(self.rank)+formatted_date+name)
                torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, f"ep={epoch}_{name}"))
        

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank, rank % num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--jobname",type=str, default='debug', help="jobname")
    parser.add_argument("--save_root",type=str,default='./output/ckpts',help="save_root")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per gpu."#112
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="resume training from checkpoint."
    )
    parser.add_argument(
        "--pca_ratio", type=float, default=0.85, help="PCA ratio."
    )    
    parser.add_argument(
        "--accum_steps", type=int, default=4, help="accumulated gradient steps."
    )    
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    args = parser.parse_args()
    rank,local_rank = setup_distributed()
    print("DDP Done")


    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank, 
        args=args
        )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length)
