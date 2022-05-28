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
    make    fold=0 \
            prefix='hard_aug_dice_ce' \
            loss_weights='1,0,1' \
            scheduler='cosine' \
            backbone='timm-efficientnet-b7' \
            epochs=30 \
            input_size='512,512' \
            num_classes=4 \
            batch_size=24 \
            pretrained=True \
            dataset='uw-gi' \
            data_dir='data/uw-gi-25d' \
            pretrained_checkpoint='' \
            model_name=${model_name} \
            use_ema=True \
            train
done
