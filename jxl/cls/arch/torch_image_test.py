from jxl.cls.arch.torch_image import *


def load_pth_tar_test():
    folder = Path('/opt/ias/project/shtm/model/cabin')
    file = folder / 'can-amount'
    net = load_pth_tar(6, file)

    net = net.module.cuda()
    print('cuda', net)

    print('\nmodel type:', type(net))
    print('state_dict:', len(net.state_dict()))


def a_test(show_state: bool):
    net = create('resnet18', 2, pretrained=True)
    print(net)

    print('\nmodel type:', type(net))
    print('state_dict:', len(net.state_dict()))

    if show_state:
        sd = net.state_dict()
        print('state_dict:', type(sd))
        for k, v in sd.items():
            print(k, "\t", v.size())


if __name__ == '__main__':
    # a_test(False)
    load_pth_tar_test()
