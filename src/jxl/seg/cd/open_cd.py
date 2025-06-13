from pathlib import Path

import numpy as np
import torch
from jvi.image.image_nda import ImageNda, ImageNdas
from jxl.seg.iseg import ISeg, ISegRes, SegOpt
from jxl.seg.mask_res import MaskRes
from mmcv import Config, ConfigDict
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
from mmseg.utils import build_dp
from mmseg.utils import setup_multi_processes
from opencd import DIEncoderDecoder
from torch import Tensor

_models = {"changer_r18": "configs/changer_ex_r18_512x512_40k_levircd.py"}


def make_cfg(cfg_file: Path, gpu_ids: list[int]) -> Config:
    """制作CFG"""
    cfg = Config.fromfile(cfg_file)
    setup_multi_processes(cfg)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.gpu_ids = gpu_ids
    cfg.model.train_cfg = None
    return cfg


def make_model(cfg: Config, device: str, model_file: Path) -> MMDataParallel:
    """制作模型"""
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    assert isinstance(model, DIEncoderDecoder)

    checkpoint: dict = load_checkpoint(model, str(model_file), map_location="cpu")
    assert isinstance(checkpoint, dict)
    meta = checkpoint.get("meta", {})

    assert "CLASSES" in meta and "PALETTE" in meta
    model.CLASSES = meta["CLASSES"]
    model.PALETTE = meta["PALETTE"]
    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    cfg.device = device
    assert torch.cuda.is_available()

    model: DIEncoderDecoder = revert_sync_batchnorm(model)  # type: ignore
    assert isinstance(model, DIEncoderDecoder)

    model1: MMDataParallel = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)  # type: ignore
    assert isinstance(model1, MMDataParallel)
    return model1


def make_pipline(img_norm_cfg: ConfigDict, shape: tuple[int, int]) -> Compose:
    """构建图像处理管道"""
    pipline_cfg = [
        {"type": "MultiImgResize", "img_scale": shape, "keep_ratio": True},
        {
            "type": "MultiImgNormalize",
            "mean": img_norm_cfg["mean"],
            "std": img_norm_cfg["std"],
            "to_rgb": img_norm_cfg["to_rgb"],
        },
        {"type": "MultiImgImageToTensor", "keys": ["img"]},
    ]
    return Compose(pipline_cfg)


class OpenCD(ISeg):
    """OpenCD Change Detector"""

    model_class = "open_cd"

    def __init__(self, model_path: Path, opt: SegOpt, device: str = "cuda"):
        super().__init__(model_path, opt, device)

        # cfg_file = Path(__file__).parent / _models['changer_r18']
        cfg_file = model_path.with_suffix(".py")

        gpu_ids = [0]  # TODO:

        cfg = make_cfg(cfg_file, gpu_ids)
        self._model = make_model(cfg, device, model_path)

        self._shape = cfg["test_pipeline"][1]["img_scale"]  # TODO: ?
        self._pipline = make_pipline(cfg["img_norm_cfg"], self._shape)

        shape3c = (*self._shape, 3)
        self._meta = {
            "ori_shape": shape3c,
            "img_shape": shape3c,
            "pad_shape": shape3c,
            "flip": False,
        }

    def forward_np(self, images: list[np.ndarray]) -> np.ndarray:
        """np输入/输出推断"""
        assert isinstance(images, list)
        assert len(images) == 2

        data_in: dict = self._pipline({"img": images})

        tensor: Tensor = data_in["img"].unsqueeze(0)  # type: ignore
        assert isinstance(tensor, Tensor)
        assert tensor.shape == torch.Size([1, 6, *self._shape])

        with torch.no_grad():
            result: list[np.ndarray] = self._model.forward(
                [tensor], [[self._meta]], return_loss=False
            )
            assert len(result) == 1
            assert isinstance(result[0], np.ndarray)
            r0 = result[0].astype(np.uint8) * 255
            assert r0.shape == self._shape
        return r0

    def forward(self, images: ImageNdas) -> ISegRes:
        """输入/输出推断"""
        mask = self.forward_np([im.data() for im in images])
        return MaskRes(ImageNda(data=mask), self._opt.min_area)
