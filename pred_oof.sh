for fold in 0 1 2 3 4 ; do
    prefix=logs/Unet/${fold}/timm-efficientnet-b5_is512,512_bs8_e30_rnd_roi_fp32/
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/main_25d.py \
                                        --model_name Unet \
                                        --backbone timm-efficientnet-b5 \
                                        --multilabel False \
                                        --resume ${prefix}/best.pth \
                                        --pred True \
                                        --fold ${fold} \
                                        --csv train_valid_case_clean.csv \
                                        --num_classes 4
done
