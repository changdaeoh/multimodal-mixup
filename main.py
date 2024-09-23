import wandb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--startlr',type=float,default=0.0,help="initial learning rate")
parser.add_argument('--lr',type=float,default=1e-6,help="initial learning rate")
parser.add_argument('--wd',type=float,default=0.2,help="weight_decay")
parser.add_argument('--bs',type=int,default=128,help="Batch size")
parser.add_argument('--epoch',type=int,default=9,help="Epoch")
parser.add_argument('--original_neg', help="use the un-mixed negative", action="store_true")
parser.add_argument('--sep_scale',  help="temperature separation",  action="store_true")
parser.add_argument('--reinit',  help="run Group",  action="store_true")
parser.add_argument('--save',type=str, default="")
parser.add_argument('--warm_r',type=float,default=0.1,help="warmup ratio")
parser.add_argument('--valid',   help="run Group",  action="store_true")
parser.add_argument('--weight',   help="load pretained weight",type=str,default=None)
parser.add_argument('--divt',type=float,default=1.0,help="Temperature Dividing Scaler")
parser.add_argument('--perc', type=float,default=1.0,help="Data Percentage")
parser.add_argument('--dataset',type=str,default="flickr",help="Dataset name. Default Flickr30k")
parser.add_argument('--seed',type=int, default=1)
parser.add_argument('--noclip',  type=int, default=0, help="disjoint model CLIP training mode")
parser.add_argument('--clip_backbone',  type=str, default='vit_b32', choices=['rn50','vit_b32','vit_b16'])

# multimodal-mixup-specific
parser.add_argument('--vmix',type=float, default=0.0)
parser.add_argument('--lmix',type=float, default=0.0)
parser.add_argument('--vlmix',type=float, default=0.0)
parser.add_argument('--mmmix',type=float, default=0.01)
parser.add_argument('--noise',type=float, default=0.0)
parser.add_argument('--beta1',type=float, default=1.0)
parser.add_argument('--beta2',type=float, default=1.0)
parser.add_argument('--betavariate',type=float, default=0.2)
parser.add_argument('--schedule',type=float)
parser.add_argument('--tau',type=float,default=0.01)
parser.add_argument('--tau2',type=float,default=0.07)

parser.add_argument('--checkpoint',type=str, default='')
parser.add_argument('--eval_only',   help="perform eval with ckpt",  action="store_true")
parser.add_argument('--pj_name',type=str, help="WB Project Name",default="mm23mmix")
parser.add_argument('--name',type=str, help="RUN Name",default="mm23mmix")
args = parser.parse_args()

run = wandb.init(project=args.pj_name,allow_val_change=True,name=args.name, entity='changdaeoh')
wandb.config.update(args,allow_val_change=True)

import os
import math
import random
import torch
import torchvision
from torchvision import datasets
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from functools import partial
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from clip import clip
from clip.model import convert_weights, CLIP, mixer_hack
import matplotlib.pyplot as plt

from utils import *
from dataset import COCODataset, FlickerDataset
import math
from loss import *
import copy
import pandas as pd
from tqdm import tqdm

DATA_PATH='YourPath'
torch.Tensor.normalize = lambda x: x/x.norm(dim=-1, keepdim=True)

IS_FIRST = True
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = args.seed

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic  = True
    torch.backends.cudnn.benchmark      = False
    np.random.seed(SEED)
    random.seed(SEED)
set_seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def convert_models_to_fp32(model): 
    for p in model.parameters():
        if p.grad is not None :
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

def convert_models_to_fp16(model): 
    for p in model.parameters():
        p.data = p.data.half() 

def sph_inter(a,b,s):
    theta = torch.acos( (a*b).sum(dim=[1] )).view(a.shape[0],1)
    n1 = torch.sin(s*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-s)*theta)/torch.sin(theta)*b
    return n1+n2

def do_train(trainloader,clip_model,optimizer,epoch,args,scheduler=None,logits_scale2=None, hist=None):
    print("training...")
    clip_model.eval()
    beta = 0.1

    temperature1 = (torch.ones([]).to(device) / args.tau)
    temperature2 = (torch.ones([]).to(device) / args.tau2)

    train_loss_acc = 0
    hard_i_history, hard_t_history, mhard_i_history, mhard_t_history = 0,0,0,0
    titer = len(trainloader)
    for batch_idx,sample in enumerate(tqdm(trainloader)):
        images, text_tok    = sample
        captions=text_tok
        global_step = epoch*titer + batch_idx
        if len(captions) != args.bs : break #Drop Last batch
        images      = images.to(device)
        if not args.noclip :
            text_tok    = clip.tokenize(text_tok,truncate=True).to(device)
        _, _,image_features,text_features = clip_model(images,text_tok)
        logits_per_image = image_features@text_features.T
        logits_per_text  = logits_per_image.T
        
        targets_orig    = torch.eye(len(captions)).to(device) 
        loss    =   torch.zeros([]).to(device)
        loss    +=  clip_loss(logits_per_image*(temperature1/args.divt),targets_orig)

        I       = targets_orig
        I_R     = torch.flip(I,dims=[0])
        I_D     = 1-I

        def write_original_neg(target,original_neg):
            cross_I = I+I_R
            cross_I_D = 1 - cross_I
            return target*cross_I + original_neg*cross_I_D

        loss_mix = torch.zeros([]).to(device)

        if epoch > -1 :
            if args.vmix :
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to("cuda:0").half()
                pos1 = sph_inter(image_features, torch.flip(image_features,dims=[0]), lamb)
                mix_logits = pos1@text_features.T
                if args.original_neg :
                    mix_logits = write_original_neg(mix_logits,logits_per_image)
                mix_logits = mix_logits*temperature1/args.divt
                loss_mix += args.vmix*calc_mix_loss(mix_logits,lamb)

            if args.lmix :
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to("cuda:0").half()
                pos1 = sph_inter(text_features, torch.flip(text_features,dims=[0]), lamb)

                mix_logits = image_features@pos1.T

                if args.original_neg :
                    mix_logits = write_original_neg(mix_logits,logits_per_image)

                mix_logits = mix_logits*temperature1/args.divt

                loss_mix += args.lmix*calc_mix_loss(mix_logits,lamb)

            if args.vlmix :
                lamb = torch.Tensor([random.betavariate(args.beta1,args.beta1)]).to("cuda:0").half()
                image_features_mixed = sph_inter(image_features, torch.flip(image_features,dims=[0]), lamb)
                text_features_mixed = sph_inter(text_features, torch.flip(text_features,dims=[0]), lamb)

                mix_logits = image_features_mixed@text_features_mixed.T

                if args.original_neg :
                    mix_logits = write_original_neg(mix_logits,logits_per_image)

                mix_logits = mix_logits*temperature1/args.divt

                loss_mix += args.vlmix*clip_loss(mix_logits,I+I_R)

            if args.mmmix :
                lamb = torch.Tensor([random.betavariate(args.beta2,args.beta2)]).to("cuda:0").half()
                targets_orig    = torch.eye(len(captions)).to(device)
                neg1 = sph_inter(image_features, text_features, lamb)
                logits_per_image2    = image_features@neg1.T
                logits_per_text2     = text_features@neg1.T

                logits_per_image2   = logits_per_image*I    +   logits_per_image2*I_D
                logits_per_text2    = logits_per_text*I     +   logits_per_text2*I_D

                if args.sep_scale:
                    logits_per_image2   = logits_per_image2*temperature2/args.divt
                    logits_per_text2    = logits_per_text2*temperature2/args.divt
                else:
                    logits_per_image2   = logits_per_image2*temperature1/args.divt
                    logits_per_text2    = logits_per_text2*temperature1/args.divt

                loss_mix           += args.mmmix*clip_loss2(logits_per_image2,I,logits_per_text2,I)

            train_loss_acc += ( loss.item() + loss_mix.item() )
            wandb.log({"train_loss_iter" : loss.item()})
            wandb.log({"train_MIX_loss_iter" : loss_mix.item()})
            wandb.log({"logit_scale" : temperature1.item()})
            wandb.log({"logit_scale2" : temperature2.item()})

        if args.schedule: loss += (1/(epoch+1))*loss_mix
        else:             loss += loss_mix

        if scheduler is not None:
            scheduler(global_step)
        optimizer.zero_grad()
        loss.backward()
        if not args.noclip:
            convert_models_to_fp32(clip_model)
        optimizer.step()
        if not args.noclip:
            convert_weights(clip_model)

inv_normalize = torchvision.transforms.Normalize(
    mean = [-0.485/0.229*255.0, -0.456/0.224*255.0, -0.406/225*255.0],
    std = [1/0.229,1/0.224,1/0.225])

def do_valid(validloader,clip_model,optimizer,args,run_calib=True,wandb_prefix="",epoch=0):
    print("Validating...")
    clip_model.eval()

    for p in clip_model.parameters():
        p.data = p.data.float() 

    with torch.no_grad():
        valid_loss_acc = 0
        tot_correct = 0
        tot_correct2 = 0
        tot_len = 0

        image_features = None
        text_features = None

        for batch_idx, sample in enumerate(tqdm(validloader)):
            images, text_tok = sample
            captions=text_tok
            images      = images.to("cuda:0")

            if not args.noclip :
                text_tok    = clip.tokenize(text_tok,truncate=True).to(device)
            _, _, image_feature, text_feature = clip_model(images,text_tok)
            if batch_idx == 0:
                image_features = image_feature
                text_features = text_feature
            else:
                image_features = torch.cat((image_features, image_feature),dim=0)
                text_features = torch.cat((text_features, text_feature),dim=0)

        temperature1 = (torch.ones([]).to(device) / args.tau)
        temperature2 = (torch.ones([]).to(device) / args.tau2)

        logits_per_image = image_features @ text_features.T 
        logits_per_text  = logits_per_image.T
        I2T = compute_metrics_pytorch(logits_per_image)
        T2I = compute_metrics_pytorch(logits_per_text)
        R1Sum = I2T["R1"] + T2I["R1"]
        print(f'==========================')
        print(f'Image2Text Retrieval : {I2T["R1"]}   {I2T["R5"]}   {I2T["R10"]}')
        print(f'Text2Image Retrieval : {T2I["R1"]}   {T2I["R5"]}   {T2I["R10"]}')
        print(f'==========================')
        I2T = add_key_prefix(I2T,"Valid_I2T_")
        T2I = add_key_prefix(T2I,"Valid_T2I_")

        alignment   = 0.0
        alignment   = rel_lalign_torch(text_features,image_features)

        mm_unif     = 0.0
        mm_unif     = lunif(torch.cat([image_features,image_features], dim=0),t=2)

        wandb.log({wandb_prefix+"Alignment" :alignment.item()} )
        wandb.log({wandb_prefix+"Uniformity":mm_unif.item()})

        logits_per_image    = logits_per_image.type(torch.float32)  
        targets_orig        = torch.eye(logits_per_image.shape[0]).to("cuda:0")
        final_loss          = clip_loss(logits_per_image,targets_orig)
        wandb.log({wandb_prefix+"valid_loss_iter" : final_loss.item()})
        valid_loss_acc      += final_loss.item()

    convert_weights(clip_model) 
    return valid_loss_acc ,I2T, T2I, R1Sum, mm_unif.item(), alignment.item()

# adapted from "https://github.com/facebookresearch/SIMAT"
def SIMAT_eval(clip_model,prep,domain='dev',args=None):
    DB_PATH = f'{DATA_PATH}/SIMAT/simat_db/images/'
    model = clip_model
    for p in model.parameters():
        p.data = p.data.float() 
    ds = datasets.ImageFolder(DB_PATH, transform=prep)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=10, shuffle=False)

    img_enc = torch.cat([(model.encode_image2(b.to(device))).cpu().detach() for b, i in tqdm(dl)]).float()

    fnames = [x[0].name for x in datasets.ImageFolder(DB_PATH, loader=Path)]
    region_ids = [int(x[:-4]) for x in fnames]

    transfos = pd.read_csv('SIMAT/simat_db/transfos.csv', index_col=0)
    words = list(set(transfos.target) | set(transfos.value))
    if not args.noclip :
        tokens = clip.tokenize(words)
        word_encs = torch.cat([(model.encode_text2(b.to(device))).cpu().detach() for b in tqdm(tokens.split(32))])
    else :
        tokens = words
        word_encs = torch.cat([(model.encode_text2(b)).cpu().detach() for b in tqdm(tokens)])

    img_enc_mapping = dict(zip(region_ids, img_enc))
    w2we = dict(zip(words, word_encs))

    emb_key = 'clip'
    output = {}
    transfos = pd.read_csv('SIMAT/simat_db/transfos.csv', index_col=0)
    triplets = pd.read_csv('SIMAT/simat_db/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))
    
    transfos = transfos[transfos.is_test == (domain == 'test')]    
    transfos_did = [rid2did[rid] for rid in transfos.region_id]
    
    clip_simat = img_enc_mapping
    img_embs_stacked = torch.stack([clip_simat[did2rid[i]] for i in range(len(clip_simat))]).float()
    img_embs_stacked = img_embs_stacked.normalize()
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])
    
    word_embs = w2we
    w2v = {k:(v.float()).normalize() for k, v in word_embs.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])
    
    oscar_scores = torch.load('SIMAT/simat_db/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)
    
    outtt = []
    for lbd in [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,7]:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs_stacked.T).topk(5).indices
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos_did)]
        
        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > .5

        output[lbd] = float(100*np.average(scores, weights=weights))
        outtt.append(float(100*np.average(scores, weights=weights)))

    print(output)
    return max(outtt)

#! for the disjoint models' CLIP training
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder,self).__init__()
        self.img_encoder = models.resnet50(pretrained=True)
        self.img_encoder.fc = nn.Identity()
    
    def forward(self,x):
        return self.img_encoder(x)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder,self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self,x):
        encoded_input = self.tokenizer(x, padding=True ,truncation=True,return_tensors='pt',max_length=200).to("cuda:0")
        output = self.model(**encoded_input)
        return output.last_hidden_state[:,0,:]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
    
    def forward(self, x):
        x = self.projection(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self, img_encoder, txt_encoder, img_emb_dim, txt_emb_dim,joint_emb_dim):
        super().__init__()
        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder
        self.img_head    = ProjectionHead(img_emb_dim,projection_dim=joint_emb_dim)
        self.txt_head    = ProjectionHead(txt_emb_dim,projection_dim=joint_emb_dim)
        self.img_emb_dim = img_emb_dim
        self.txt_emb_dim = txt_emb_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image2(self,img):
        img_feature = self.img_encoder(img)
        img_embed   = self.img_head(img_feature)
        img_embed = img_embed / img_embed.norm(dim=-1,keepdim=True)
        return img_embed

    def encode_text2(self,txt):
        txt_feature = self.txt_encoder(txt)
        txt_embed   = self.txt_head(txt_feature)
        txt_embed = txt_embed / txt_embed.norm(dim=-1,keepdim=True)
        return txt_embed

    def forward(self, img,txt):
        img_embed = self.encode_image2(img)
        txt_embed = self.encode_text2(txt)
        logits_per_image = img_embed@txt_embed.T
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text,img_embed,txt_embed
    

if args.clip_backbone == 'vit_b32':
    bbone = 'ViT-B/32'
elif args.clip_backbone == 'vit_b16':
    bbone = 'ViT-B/16'
elif args.clip_backbone == 'rn50':
    bbone = 'RN50'
else:
    pass
clip_model, preprocess = clip.load(bbone, device=device,jit=False,use_shared=wandb.config.shared, prompts_length = 0)
if args.checkpoint is not None:
    if(os.path.isfile(args.checkpoint)):
        checkpoint = torch.load(args.checkpoint, map_location = "cpu")
        start_epoch = checkpoint["epoch"]
        state_dict = checkpoint["state_dict"]
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        clip_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint '{args.checkpoint}' (start epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at {args.checkpoint}")

if args.noclip :
    img_encoder = ImageEncoder()
    txt_encoder = TextEncoder()
    clip_model = CLIPModel(img_encoder, txt_encoder, 2048, 768, 256).cuda()

if args.weight :
    clip_model.load_state_dict(torch.load(args.weight)['model_state_dict'])

if wandb.config.reinit :
    clip_model.initialize_parameters()

if args.dataset == "flickr" :
    trainset    = FlickerDataset(f'{DATA_PATH}/flickr30k',transform=preprocess,perc=args.perc).filter_df("train")
    validset    = FlickerDataset(f'{DATA_PATH}/flickr30k',transform=preprocess).filter_df("valid")
    testset     = FlickerDataset(f'{DATA_PATH}/flickr30k',transform=preprocess).filter_df("test")
elif args.dataset == "coco" :
    trainset    = COCODataset(f'{DATA_PATH}/coco/images/train2017',anon_path=f'{DATA_PATH}/coco/images/annotations/captions_train2017.json',transform=preprocess,perc=args.perc)
    validset    = COCODataset(f'{DATA_PATH}/coco/images/val2017',anon_path=f'{DATA_PATH}/coco/images/annotations/captions_val2017.json',transform=preprocess)
    testset     = COCODataset(f'{DATA_PATH}/coco/images/val2017',anon_path=f'{DATA_PATH}/coco/images/annotations/captions_val2017.json',transform=preprocess)
else:
    raise ValueError

trainloader = DataLoader(trainset, batch_size= wandb.config.bs, shuffle=True,num_workers=8,worker_init_fn=seed_worker)
print("# of train samples : " , len(trainset))
validloader = DataLoader(validset, batch_size= wandb.config.bs, shuffle=False,worker_init_fn=seed_worker)
testloader  = DataLoader(testset, batch_size= wandb.config.bs, shuffle=False,worker_init_fn=seed_worker)
testloader2  = DataLoader(testset, batch_size= 128, shuffle=False,worker_init_fn=seed_worker)

print("# of batch: ",len(trainloader))
tot_iters = len(trainloader) * args.epoch
optimizer = torch.optim.AdamW( list(clip_model.parameters()) ,lr=args.lr,weight_decay=args.wd)
scheduler = cosine_lr(optimizer, args.lr, tot_iters * args.warm_r, tot_iters)

# Train / Eval loop
Best_R1_sum = -1
best_ep = 0
best_i2t = 0
best_t2i = 0
epoch = 0
best_u = 0
best_a = 0
if args.eval_only:
    current_best_val,I2T,T2I,R1Sum, unif, align = do_valid(testloader,clip_model,optimizer,args=args,epoch=epoch)
    if R1Sum > Best_R1_sum : Best_R1_sum = R1Sum; best_ep = epoch; best_i2t = I2T; best_t2i = T2I
    wandb.log(I2T)
    wandb.log(T2I)
    wandb.log({"R1_Sum":R1Sum})
    wandb.log({"Best_R1_Sum" : Best_R1_sum, 'BEST-EP':best_ep, 'BEST_I2T':best_i2t, 'BEST_T2I':best_t2i, 'BEST_U':best_u, 'BEST_A':best_a})
else:
    for epoch in range(wandb.config.epoch):
        if args.startlr:
            if epoch == 0:
                optimizer.param_groups[0]['lr'] = args.startlr 
            elif epoch == 1:
                optimizer.param_groups[0]['lr'] = args.lr

        print("EPOCH ",epoch)
        max_simat = 0.0
        do_train(trainloader,clip_model,optimizer,epoch=epoch,args=args, scheduler=scheduler, hist=None)
        if (epoch + 1) == args.epoch:
            current_best_val,I2T,T2I,R1Sum, unif, align = do_valid(testloader,clip_model,optimizer,args=args,epoch=epoch)
            max_simat = SIMAT_eval(clip_model,preprocess, args=args)
            wandb.log({"max_simat":max_simat})
            if R1Sum > Best_R1_sum : Best_R1_sum = R1Sum; best_ep = epoch; best_i2t = I2T; best_t2i = T2I; best_u= unif; best_a =align
            wandb.log(I2T)
            wandb.log(T2I)
            wandb.log({"R1_Sum":R1Sum})

    wandb.log({"Best_R1_Sum" : Best_R1_sum, 'BEST-EP':best_ep, 'BEST_I2T':best_i2t, 'BEST_T2I':best_t2i, 'BEST_U':best_u, 'BEST_A':best_a})

if args.save:
    torch.save({'model_state_dict':clip_model.state_dict()}, 
            os.path.join(args.save, args.name+'.pt'))