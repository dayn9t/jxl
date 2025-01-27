import torch

from utils.general import box_iou  # type: ignore


def test_iou():
    x1 = torch.tensor([
        [0.0, 0, 1000, 1000],
        [1000, 1000, 2000, 2000],
    ])

    x2 = torch.tensor([
        [500.0, 500, 1500, 1500],
        # [1500, 1500, 2500, 2500],
    ])

    print('iou:', x1, x1.dtype)

    x3 = box_iou(x1, x2)

    print('iou:', x3)
    # print('box2:', box2)

    pre_det = torch.zeros(1, 6, dtype=torch.float)
    print('iou:', pre_det, pre_det.dtype)

    t = x3 > 0.1
    print('t:', t, pre_det.dtype)
    # t = False
    # assert t == True
