bg=./data/bg/
fold=0
backbone=efficientnet-b0
batch_size=32
input_size=320,384
epochs=15
NCCL_P2P_DISABLE=0
output_dir=./logs/25D/Unet/${backbone}_is${input_size}_bs${batch_size}_e${epochs}_bndist_lr2e3

train:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} PYTHONPATH=. \
	python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 2106 scripts/main_25d.py \
	--csv train_valid.csv \
	--fold ${fold} \
	--num_classes 3 \
	--backbone ${backbone} \
	--output_dir ${output_dir} \
	--batch_size_per_gpu ${batch_size} \
	--input_size ${input_size} \
	--epochs ${epochs} \
	--use_fp16 False

