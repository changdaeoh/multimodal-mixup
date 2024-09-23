import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import torch
from dosnes import dosnes as dosnes_pkg


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster

def dosnes(image_embed, text_embed, name="train"):
    img_np = image_embed.float().detach().cpu().numpy()
    txt_np = text_embed.float().detach().cpu().numpy()
    all_emb = np.concatenate([img_np,txt_np])

    metric = "sqeuclidean"
    momentum = 0.1
    final_momentum = 0.7
    mom_switch_iter = 250
    max_iter = 1000
    learning_rate = 400
    min_gain = 0.01
    model = dosnes_pkg.DOSNES(momentum = momentum, final_momentum = final_momentum, learning_rate = learning_rate, min_gain = min_gain,max_iter = 1000, verbose_freq = 10, metric = metric, verbose = 1, random_state=0)

    len_img = img_np.shape[0]
    X = all_emb
    y = np.concatenate( ( np.ones(shape=(len_img,)) , np.zeros(shape=(len_img,)) ) ,dtype=np.float32)
    
    aaa = model.fit_transform(X, y, filename="training.gif")

    np.save(f'{name}_dosnes_vec_{len_img}.npy',aaa)
    plt.clf()
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(aaa[:len_img,0],aaa[:len_img,1],aaa[:len_img,2],c='#FFC514',alpha=0.6,edgecolors='white')
    ax.scatter(aaa[len_img:,0],aaa[len_img:,1],aaa[len_img:,2],c='#00C97A',alpha=0.6,edgecolors='white')
    plt.savefig("ffig.pdf",pad_inches = 0)
    wandb.log({"tsne"+name:wandb.Image(plt)})
    return plt

def tsne_vec_3d(image_embed,text_embed,name="train"):
    img_np = image_embed.float().detach().cpu().numpy()
    txt_np = text_embed.float().detach().cpu().numpy()
    all_emb = np.concatenate([img_np,txt_np])
    aaa = TSNE(n_components=3,init='random',n_jobs=1).fit_transform(all_emb)
    len_img = img_np.shape[0]

    plt.clf()
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(aaa[:len_img,0],aaa[:len_img,1],aaa[:len_img,2],c='#FFC514',alpha=0.6,edgecolors='white')
    ax.scatter(aaa[len_img:,0],aaa[len_img:,1],aaa[len_img:,2],c='#00C97A',alpha=0.6,edgecolors='white')
    plt.savefig("ffig.pdf",pad_inches = 0)
    wandb.log({"tsne"+name:wandb.Image(plt)})

    return plt

def tsne_vec(image_embed,text_embed,name="train"):
    img_np = image_embed.float().detach().cpu().numpy()
    txt_np = text_embed.float().detach().cpu().numpy()
    all_emb = np.concatenate([img_np,txt_np])
    aaa = TSNE(n_components=2,init='random',n_jobs=1).fit_transform(all_emb)
    bbb = LocallyLinearEmbedding(n_components=2).fit_transform(all_emb)
    len_img = img_np.shape[0]

    plt.clf()
    #plt.show()
    sns.set_style("ticks")
    plt.scatter(aaa[:len_img,0],aaa[:len_img,1],c='#FFC514',alpha=0.6,edgecolors='white')
    plt.scatter(aaa[len_img:,0],aaa[len_img:,1],c='#00C97A',alpha=0.6,edgecolors='white')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.savefig("ffig.pdf",pad_inches = 0)
    wandb.log({"tsne"+name:wandb.Image(plt)})

    return plt

import pdb
def compute_metrics(logits):
    #pdb.set_trace()
    sorted_logits = np.sort(-logits, axis=1)
    pos_logits = np.diag(-logits)
    pos_logits = pos_logits[:, np.newaxis]
    idx = sorted_logits - pos_logits
    idx = np.where(idx == 0)[1]
    metrics = {}
    try:
        print(1.0 / len(idx))
    except:
        pdb.set_trace()
        print('logits',logits)
        print('sorted_logits',sorted_logits)
        print('pos_logits',pos_logits)
        print('idx',idx)
    metrics['R1'] = float(np.sum(idx == 0)) * 100 / len(idx)
    metrics['R5'] = float(np.sum(idx < 5)) * 100 / len(idx)
    metrics['R10'] = float(np.sum(idx < 10)) * 100 / len(idx)
    metrics['MR'] = np.median(idx) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(idx) + 1
    metrics["cols"] = [int(i) for i in list(idx)]
    return metrics


def compute_metrics_pytorch(logits):
    sorted_logits = torch.sort(-logits, dim=1).values
    pos_logits = torch.diag(-logits)
    pos_logits = pos_logits.unsqueeze(1)
    idx = sorted_logits - pos_logits
    idx = torch.where(idx == 0)[1]
    metrics = {}
    try:
        print(1.0 / len(idx))
    except:
        print('logits',logits)
        print('sorted_logits',sorted_logits)
        print('pos_logits',pos_logits)
        print('idx',idx)
    metrics['R1'] = float(torch.sum(idx == 0)) * 100 / len(idx)
    metrics['R5'] = float(torch.sum(idx < 5)) * 100 / len(idx)
    metrics['R10'] = float(torch.sum(idx < 10)) * 100 / len(idx)
    
    return metrics


def add_key_prefix(dic,pre):
    return {pre+k : v for k,v in dic.items()}
