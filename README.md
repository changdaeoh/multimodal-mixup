# Geodesic Multi-Modal Mixup for Robust Fine-Tuning (NeurIPS 2023)
>[Changdae Oh](https://changdaeoh.github.io/)\*, [Junhyuk So](https://github.com/junhyukso)\*, [YongTaek Lim](https://github.com/teang1995), [Hoyoon Byun](https://scholar.google.com/citations?user=55yqBlMAAAAJ&hl=en), [Minchul Shin](https://scholar.google.com/citations?user=52NtRk8AAAAJ&hl=en), [Jong-June Jeon](https://scholar.google.co.kr/citations?user=A-E3uEMAAAAJ&hl=ko), and [Kyungwoo Song](https://scholar.google.com/citations?user=HWxRii4AAAAJ&hl=ko)

## Main experiment with OpenAI CLIP checkpoints
* prepare Flickr30k, MSCOCO (2017 ver), and SIMAT datasets in `data/`
* refer `scripts/m2mix.sh` and `scripts/m3mix.sh` script files to reproduce our experiments

## Preview for OpenCLIP experiment: ClipLoss Class in `OpenCLIP>src>open_clip>loss.py`
https://github.com/mlfoundations/open_clip

```python
def sph_inter(a,b,s):
    theta = torch.acos( (a*b).sum(dim=[1] )).view(a.shape[0],1)
    n1 = torch.sin(s*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-s)*theta)/torch.sin(theta)*b
    return n1+n2

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            unimix=0.0,
            vlmix=0.0,
            mmix=0.0,
            beta_u=0.5,
            beta_m=0.5,
            m_tau=0.01
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        # multimodalmixup
        self.unimix=unimix
        self.vlmix=vlmix
        self.mmix=mmix
        self.m_tau=m_tau
        self.beta_u=beta_u
        self.beta_m=beta_m
        random.seed(1)

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        #! vanilla CL
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        #! -------------------------
        #! CL with multi-modal mixup
        #! -------------------------
        I = torch.eye(image_features.shape[0]).to("cuda:0")
        I_D = 1 - I
        if self.mmix > 0:
            lamb = torch.Tensor([random.betavariate(self.beta_m,self.beta_m)]).to("cuda:0")
            mixed_neg = sph_inter(image_features, text_features, lamb)
            logits_per_image_mm    = self.m_tau * image_features @ mixed_neg.T
            logits_per_text_mm     = self.m_tau * text_features @ mixed_neg.T
            logits_per_image_mm    = logits_per_image*I    +   logits_per_image_mm*I_D
            logits_per_text_mm     = logits_per_text*I     +   logits_per_text_mm*I_D
            mmix_loss = (
                F.cross_entropy(logits_per_image_mm, labels) +
                F.cross_entropy(logits_per_text_mm, labels)
            ) / 2

            total_loss += self.mmix * mmix_loss
            
        return {"contrastive_loss": total_loss} if output_dict else total_loss
```