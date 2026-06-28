from pathlib import Path

from loguru import logger

from jxl.cls.arch.torch_image import create, load_pth_tar


def load_pth_tar_test() -> None:
    folder = Path("/opt/ias/project/shtm/model/cabin")
    file = folder / "can-amount"
    net = load_pth_tar(6, file)

    net = net.cuda()
    logger.info("cuda {}", net)

    logger.info("model type: {}", type(net))
    logger.info("state_dict: {}", len(net.state_dict()))


def a_test(show_state: bool) -> None:
    net = create("resnet18", 2, pretrained=True)
    logger.info("{}", net)

    logger.info("model type: {}", type(net))
    logger.info("state_dict: {}", len(net.state_dict()))

    if show_state:
        sd = net.state_dict()
        logger.info("state_dict: {}", type(sd))
        for k, v in sd.items():
            logger.info("{} \t {}", k, v.size())


if __name__ == "__main__":
    # a_test(False)
    load_pth_tar_test()
