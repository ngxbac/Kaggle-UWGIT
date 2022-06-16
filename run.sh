# for i in 0 ; do
#     make    fold=${i} \
#             prefix='splitrnd' \
#             loss_weights='1,1,1' \
#             scheduler='cosine' \
#             backbone='timm-efficientnet-b5' \
#             epochs=30 \
#             input_size='768,768' \
#             num_classes=4 \
#             batch_size=16 \
#             model_name='Linknet' \
#             use_ema=False \
#             train
# done


prefix=./logs_multistages/FPN/0/timm-efficientnet-b5_is512,512_bs32_e20_stage_1

for model_name in UnetPlusPlus ; do
    for fold in 0 1 2 3 4 ; do
        make    fold=${fold} \
                csv='train_valid_case.csv' \
                prefix='stage_1' \
                log_prefix='logs_full' \
                loss_weights='1,0,1' \
                scheduler='cosine' \
                backbone='timm-efficientnet-b5' \
                epochs=30 \
                input_size='512,512' \
                num_classes=3 \
                batch_size=24 \
                lr=1e-3 \
                min_lr=0 \
                pretrained=True \
                dataset='uw-gi' \
                data_dir='data/uw-gi-25d' \
                pretrained_checkpoint='no' \
                mmcfg='mmconfigs/segformer_mit_b3.py' \
                model_name=${model_name} \
                use_ema=True \
                multilabel=True \
                train
    done
done

# for model_name in FPN ; do
#     for fold in 0 ; do
#         python scripts/avg_checkpoints.py \
#             --input logs_multistages/FPN/${fold}/timm-efficientnet-b5_is512,512_bs32_e20_stage_1/ \
#             --filter "checkpoint00*.pth" \
#             --output logs_multistages/FPN/${fold}/timm-efficientnet-b5_is512,512_bs32_e20_stage_1/sw_ema.pth \
#             --no-sort -n 5
#     done
# done


# for model_name in FPN ; do
#     for fold in 0 ; do
#         make    fold=${fold} \
#                 csv='csv/Unet_keep_0.5.csv' \
#                 prefix='stage_2' \
#                 log_prefix='logs_multistages' \
#                 loss_weights='1,0,0' \
#                 scheduler='onecycle' \
#                 backbone='timm-efficientnet-b5' \
#                 epochs=10 \
#                 input_size='512,512' \
#                 num_classes=3 \
#                 batch_size=32 \
#                 lr=2e-4 \
#                 min_lr=1e-6 \
#                 pretrained=True \
#                 dataset='uw-gi' \
#                 data_dir='data/uw-gi-25d' \
#                 pretrained_checkpoint=logs_multistages/FPN/${fold}/timm-efficientnet-b5_is512,512_bs32_e20_stage_1/sw_ema.pth  \
#                 mmcfg='mmconfigs/segformer_mit_b3.py' \
#                 model_name=${model_name} \
#                 use_ema=False \
#                 multilabel=True \
#                 train
#     done
# done


# for model_name in FPN ; do
#     for fold in 0 ; do
#         python scripts/avg_checkpoints.py \
#             --input logs_multistages/FPN/${fold}/timm-efficientnet-b5_is512,512_bs32_e20_stage_2/ \
#             --filter "checkpoint00*.pth" \
#             --output logs_multistages/FPN/${fold}/timm-efficientnet-b5_is512,512_bs32_e20_stage_2/swa.pth \
#             --no-use-ema \
#             --no-sort -n 5
#     done
# done


