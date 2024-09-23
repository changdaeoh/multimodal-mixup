#!/bin/bash
cd ..

sche=1.0
bbone='vit_b32'
ds=flickr
buni=0.0
unimix=0.0
vlmix=0.0

for beta2 in 0.5
do
for mmmix in 0.05 0.01 0.005 0.001
do
for lr in 1e-5 7e-6
do
for wd in 0.2 0.01
do

CUDA_VISIBLE_DEVICES=4 python main.py \
--pj_name YOURS --epoch 9 --tau 0.01 --schedule $sche \
--out_img --out_txt --manifold_all --neg_mix --dataset $ds \
--betavariate $buni --beta1 $buni --beta2 $beta2 \
--vmix $unimix --lmix $unimix --vlmix $vlmix --mmmix $mmmix \
--lr $lr --wd $wd --clip_backbone $bbone --sep_scale \
--name m2mix_sweep

done
done
done
done