for model_name in FPN ; do
    for fold in 0 1 2 3 4 ; do
        make    fold=${fold} \
                csv='csv/full_with_pseudo.csv' \
                prefix='v0_fix' \
                log_prefix='logs_finetune_v0' \
                loss_weights='1,1,1' \
                scheduler='cosine' \
                backbone='timm-efficientnet-b5' \
                epochs=10 \
                lr=5e-4 \
                input_size='512,512' \
                num_classes=4 \
                batch_size=16 \
                pretrained=True \
                pretrained_checkpoint=logs_clean_v0/FPN/${fold}/timm-efficientnet-b5_is512,512_bs16_e30_fix/best.pth \
                dataset='uw-gi' \
                data_dir='data/uw-gi-25d' \
                model_name=${model_name} \
                use_ema=False \
                train
    done
done


for model_name in FPN ; do
    for fold in 0 1 2 3 4 ; do
        make    fold=${fold} \
                csv='csv/full_with_pseudo.csv' \
                prefix='v0_fix' \
                log_prefix='logs_no_finetune_v0' \
                loss_weights='1,1,1' \
                scheduler='cosine' \
                backbone='timm-efficientnet-b5' \
                epochs=30 \
                lr=1e-3 \
                input_size='512,512' \
                num_classes=4 \
                batch_size=16 \
                pretrained=True \
                dataset='uw-gi' \
                data_dir='data/uw-gi-25d' \
                model_name=${model_name} \
                use_ema=False \
                train
    done
done
