fold=0
backbone=timm-efficientnet-b5
batch_size=32
input_size=512,512
epochs=30
NCCL_P2P_DISABLE=0
prefix=''
resume=''
loss_weights=''
scheduler='onecycle'
lr=1e-4
min_lr=0
num_classes=4
use_ema=False
model_name='FPN'
multilabel=False
dataset='uw-gi'
data_dir='data/uw-gi-25d'
pretrained=True
pretrained_checkpoint=''
log_prefix='logs'
pred=False
csv=train_valid_case_clean.csv
output_dir=./${log_prefix}/${model_name}/${backbone}_is${input_size}_bs${batch_size}_e${epochs}_${prefix}_${fold}
num_gpus=`nvidia-smi --list-gpus | wc -l`
distributed=0
mmcfg=mmconfigs/segformer_mit_b0.py

# if [[ ${distributed} -eq 0 ]]
# then
# 	command=python
# else
# fi

command=python -u -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port 2106

train:
	PYTHONPATH=. \
	${command} scripts/main_25d.py \
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
	--mmcfg ${mmcfg} \
	--use_fp16 True

valid:
	PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/main_25d.py \
	--csv train_valid_case_clean.csv \
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
	--lr ${lr} \
	--loss_weights ${loss_weights} \
	--scheduler ${scheduler} \
	--pred ${pred} \
	--use_fp16 True




roi_size=64
batch_size_3d=1
epochs_3d=200
space=1.5
model=segresnet
num_samples=4
out_channels=4
resume=''
lr=0.001
data_dir=""
logdir=logs/3d/yw/${model}_${roi_size}_${fold}_${epochs_3d}ep_rndcrop

train_3d:
	NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} PYTHONPATH=. \
	python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 2106 scripts/main_3d_yw.py \
    --feature_size=32 \
    --batch_size=1 \
    --roi_x ${roi_size} \
    --roi_y ${roi_size} \
    --roi_z 80 \
    --space_x ${space} \
    --space_y ${space} \
    --space_z ${space} \
    --model_name ${model} \
    --fold ${fold} \
    --out_channels ${out_channels} \
	--num_samples ${num_samples} \
    --infer_overlap=0.5 \
    --data_dir=${data_dir} \
	--output_dir ${logdir} \
	--batch_size_per_gpu ${batch_size_3d} \
	--epochs ${epochs_3d} \
	--res_block \
	--resume ${resume} \
	--lr ${lr} \
	--use_fp16 True
