#!/bin/bash

# Parameters
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test
#SBATCH --ntasks-per-node=1
#SBATCH --partition=agpu72
#SBATCH --nodelist=c[2009,2108,2109,2110]

export MASTER_PORT=12340
export WORLD_SIZE=4


### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


module load cuda/11.1
source ~/ondemand/miniconda3/etc/profile.d/conda.sh
conda activate torch1.8


# python
fold=2
backbone=timm-efficientnet-b5
batch_size=16
input_size=512,512
epochs=30
prefix='fp32'
resume='no'
loss_weights='1,0,1'
scheduler='cosine'
lr=1e-3
min_lr=0
num_classes=3
use_ema=True
model_name='FPN'
multilabel=True
dataset='uw-gi'
data_dir='data/uw-gi-25d'
pretrained=True
pretrained_checkpoint='no'
log_prefix='logs_full'
pred=False
csv=train_valid_case.csv
output_dir=./${log_prefix}/${model_name}/${backbone}_is${input_size}_bs${batch_size}_e${epochs}_${prefix}_${fold}
mmcfg=mmconfigs/segformer_mit_b0.py


srun python -u scripts/main_25d.py \
	--csv ${csv} \
	--fold ${fold} \
	--use_ema ${use_ema} \
	--model_name ${model_name} \
	--multilabel ${multilabel} \
	--data_dir ${data_dir} \
	--dataset ${dataset} \
	--num_classes ${num_classes} \
	--model_name ${model_name} \
	--backbone ${backbone} \
	--output_dir ${output_dir} \
	--batch_size_per_gpu ${batch_size} \
	--input_size ${input_size} \
	--epochs ${epochs} \
	--resume ${resume} \
	--pretrained ${pretrained} \
	--pretrained_checkpoint ${pretrained_checkpoint} \
	--lr ${lr} \
	--min_lr ${min_lr} \
	--loss_weights ${loss_weights} \
	--scheduler ${scheduler} \
	--pred ${pred} \
	--use_fp16 False

