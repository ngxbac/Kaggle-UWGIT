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


for model_name in FPN ; do
    for fold in 0 ; do
        make    fold=${fold} \
                csv='csv/Unet_keep_0.5.csv' \
                prefix='baseline_hard_aug_ema' \
                log_prefix='logs_clean_3s' \
                loss_weights='1,0,1' \
                scheduler='cosine' \
                backbone='timm-efficientnet-b5' \
                epochs=30 \
                input_size='512,512' \
                num_classes=3 \
                batch_size=16 \
                lr=1e-3 \
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
