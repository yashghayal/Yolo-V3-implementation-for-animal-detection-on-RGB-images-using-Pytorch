
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from model import YOLOv3
from utils import get_loaders, cells_to_bboxes, non_max_suppression, plot_image

def test(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, e = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, i)


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/testcustom.csv"
    )
    checkpoint = torch.load('checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    
    test(model, test_loader, 0.6, 0.5, scaled_anchors)

    print('done')

main()
