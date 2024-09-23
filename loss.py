
import math
import torch
from torch import nn

def rel_lalign_torch(x,y,alpha=2):
    pos_align = (x-y).norm(dim=1).pow(alpha)
    
    diff_tensor = (x.unsqueeze(1) - y.unsqueeze(0)).view(x.shape[0], x.shape[0], x.shape[1])  # N*N*D element-wise diff
    diff_tensor = diff_tensor.norm(dim=2).pow(alpha)
    diff_tensor.fill_diagonal_(10000.0)
    # comp with top 1 neg
    neg_align = diff_tensor.min(dim=1).values
    relalign = (pos_align - neg_align).mean()

    return relalign

def rel_lalign(x,y,alpha=2, neighbor=1, relative='ratio', save_name='relative_hist',ep=0):
    x = x.cpu()
    y = y.cpu()
    pos_align = (x-y).norm(dim=1).pow(alpha)
    #import pdb; pdb.set_trace()
    diff_tensor = (x.unsqueeze(1) - y.unsqueeze(0)).view(x.shape[0], x.shape[0], x.shape[1])  # N*N*D element-wise diff
    diff_tensor = diff_tensor.norm(dim=2).pow(alpha)         # N*N diffence matrix(aggregated by feature dim)
    diff_tensor.fill_diagonal_(10000.0)                      # eliminate diagonal(pos.pair) elements
    if neighbor == 1:        # top 1
        neg_align = diff_tensor.min(dim=1).values
    elif neighbor == -1:     # total mean negative
        neg_align = diff_tensor.mean(dim=1)
    else:                    # top k
        neg_align = torch.topk(diff_tensor, dim=1, k=neighbor).values.mean(dim=1)

    if relative == 'ratio':
        relalign = (pos_align / neg_align).mean()
    else:
        relalign = (pos_align - neg_align).mean()

    return relalign

def lalign(x,y,alpha=2):
    return (x-y).norm(dim=1).pow(alpha).mean()

def lunif(x,t=2):
    sq_pdist = torch.pdist(x.float(),p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def clip_loss(logits,targets):
    img_loss = cross_entropy(logits,targets)
    txt_loss = cross_entropy(logits.T,targets.T)
    final_loss = ((img_loss + txt_loss)/2.0 ).mean()
    return final_loss
 
def clip_loss2(logits1,targets,logits2=None,targets2=None):
    loss = cross_entropy(logits1,targets).mean()
    
    if logits2 != None and targets2 != None :
        loss += cross_entropy(logits2,targets).mean()

    return loss


def calc_mix_loss(logits_mix,lamb,mode="mix",c=-1,l=1):
    matrix_sz = logits_mix.shape[0]
    if c>0 :
        lamb1 = math.sqrt(  lamb**2       + c**2    )
        lamb2 = math.sqrt(  (1-lamb)**2   + c**2    )
        lamb1,lamb2 = lamb1/(lamb1+lamb2) , lamb2/(lamb1+lamb2)
    else :
        lamb1 = lamb
        lamb2 = 1-lamb
    if mode == "mix":
        targets_mix = torch.eye(matrix_sz).to("cuda:0")*lamb1  +  (lamb2)*torch.flip(torch.eye(matrix_sz).to("cuda:0"),dims=[0])
    elif mode == "temp":
        targets_mix = torch.eye(matrix_sz).to("cuda:0")
        logits_mix  = logits_mix*lamb
    targets_mix = targets_mix.to("cuda:0")
    if l==1 :
        return clip_loss(logits_mix, targets_mix)
    if l==2 :
        return clip_loss2(logits_mix, targets_mix)

#! Hard-Negative Contrastive Learning
def hcl_clip_loss(emb1, emb2, tau_plus=0.1, beta=1.0):
    N = emb1.shape[0]
    neg = torch.exp(torch.mm(emb1, emb2.t().contiguous()) * 2)
    old_neg = neg.clone()
    negative_mask = torch.ones((N, N)) - torch.eye(N)
    mask = negative_mask.bool().to(emb1.device)
    neg = neg.masked_select(mask).view(N, -1)
    pos = torch.exp(torch.sum(emb1 * emb2, dim=-1) * 2)
    imp = (beta*neg.log()).exp()
    reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
    Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss