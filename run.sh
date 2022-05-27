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
            prefix='pretrained_abdomen' \
            loss_weights='1,1,1' \
            scheduler='cosine' \
            backbone='timm-efficientnet-b5' \
            epochs=30 \
            input_size='512,512' \
            num_classes=4 \
            batch_size=32 \
            pretrained=True \
            dataset='uw-gi' \
            data_dir='data/uw-gi-25d' \
            pretrained_checkpoint='logs/pretrained_abdomen/FPN/0/timm-efficientnet-b5_is512,512_bs32_e15_abdomen/checkpoint.pth' \
            model_name=${model_name} \
            use_ema=False \
            train
done
