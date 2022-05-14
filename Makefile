bg=./data/bg/
fold=0
backbone=efficientnet-b0
batch_size=32
input_size=320,384
epochs=30
NCCL_P2P_DISABLE=0
output_dir=./logs/25D/Unet/${backbone}_is${input_size}_bs${batch_size}_e${epochs}_bndist_lr2e3_scale

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


roi_size=80
batch_size_3d=1
epochs_3d=300
space=1.5
model=segresnet
num_samples=32
logdir=logs/3d/pilot/${model}_${roi_size}_${space}_${fold}_scale_intensity_${num_samples}samples_${epochs_3d}ep

train_3d:
	CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} PYTHONPATH=. \
	python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 2106 scripts/main_3d.py \
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
    --out_channels 3 \
	--num_samples ${num_samples} \
    --infer_overlap=0.5 \
    --data_dir=data/nii-data-2 \
	--output_dir ${logdir} \
	--batch_size_per_gpu ${batch_size_3d} \
	--epochs ${epochs_3d} \
	--use_fp16 False	
