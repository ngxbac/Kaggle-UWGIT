import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import math

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train model 2.5D', add_help=False)

    # * Model
    # dataset parameters
    parser.add_argument('--csv', default='train_valid_case_clean.csv')
    parser.add_argument('--model_name', default='Unet')
    parser.add_argument('--prefix', default='timm-efficientnet-b5_is512,512_bs8_e30_rnd_roi_fp32')
    parser.add_argument('--data_dir', default='data/uw-gi-25d/')
    parser.add_argument('--threshold', default=0.5, type=float)
    return parser


def load_mask(path):
    path = f"{data_dir}/{path}"
    mask = np.load(path)
    mask = mask.max(axis=0)
    return mask

def load_image(path):
    path = f"{data_dir}/{path}"
    path = path.replace("_mask", "_image")
    mask = np.load(path)
    mask = mask / mask.max()
    return mask

def dice(x, y, epsilon=1e-8):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    x_sum = np.sum(np.sum(np.sum(x)))

    return (2 * intersect + epsilon) / (x_sum + y_sum + epsilon)

def dice_score(pred, gt):
    h, w = pred.shape[:2]
    gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    all_scores = []
    for i in range(1, 4):
        pred_ = pred == i
        gt_ = gt == i
        score = dice(pred_, gt_)
        all_scores.append(score)

    return np.mean(all_scores)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train 2.5D', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    df = pd.read_csv(args.csv)

    keep_dfs = []
    remove_dfs = []
    for fold in range(5):
        fold_df = df[df.fold == fold].reset_index(drop=True)
        masks = fold_df['mask'].values
        preds = np.load(f"logs/{args.model_name}/{fold}/{args.prefix}/preds.npy")
        n_preds = len(preds)

        all_scores = []
        for i in tqdm(range(n_preds)):
            pred = preds[i]
            mask = masks[i]
            mask = load_mask(mask)
            score = dice_score(pred, mask)
            all_scores.append(score)

        all_scores = np.array(all_scores)
        print(all_scores)
        print("Mean dice score: ", np.mean(all_scores))
        keep_idx = np.where(all_scores > args.threshold)[0]
        remove_idx = np.where(all_scores <= args.threshold)[0]

        num_remove = len(fold_df) - len(keep_idx)
        print(f"Remove: {num_remove}")

        keep_df = fold_df.iloc[keep_idx].reset_index(drop=True)
        remove_df = fold_df.iloc[remove_idx].reset_index(drop=True)
        keep_dfs.append(keep_df)
        remove_dfs.append(remove_df)

    keep_dfs = pd.concat(keep_dfs, axis=0)
    remove_dfs = pd.concat(remove_dfs, axis=0)
    keep_dfs.to_csv(f"csv/{args.model_name}_keep_{args.threshold}.csv", index=False)
    remove_dfs.to_csv(f"csv/{args.model_name}_remove_{args.threshold}.csv", index=False)


