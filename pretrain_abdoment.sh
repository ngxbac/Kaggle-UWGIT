for model_name in FPN ; do
    make    fold=0 \
            prefix='abdomen_2' \
            loss_weights='1,1,1' \
            scheduler='cosine' \
            backbone='timm-efficientnet-b5' \
            epochs=30 \
            input_size='512,512' \
            num_classes=5 \
            batch_size=32 \
            pretrained=True \
            model_name=${model_name} \
            dataset='abdomen' \
            data_dir='data/AbdomenCT/Subtask1_2d,data/AbdomenCT/Subtask2_2d' \
            use_ema=False \
            train
done
