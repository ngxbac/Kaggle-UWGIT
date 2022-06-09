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
                prefix='sce_loss_att' \
                log_prefix='logs_clean_3s' \
                loss_weights='1,1,1' \
                scheduler='cosine' \
                backbone='tu-ecaresnet50t' \
                epochs=30 \
                input_size='320,320' \
                num_classes=4 \
                batch_size=16 \
                pretrained=True \
                dataset='uw-gi' \
                data_dir='data/uw-gi-25d' \
                pretrained_checkpoint='no' \
                model_name=${model_name} \
                use_ema=False \
                train
    done
done
