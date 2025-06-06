import torch
import torch.nn as nn
import sys
from torchvision import models

# Import CIFAR and STL-10 model definitions
from models.cifar_models import (
    cifar10_mobilenetv2_x0_5, cifar10_resnet20, cifar10_resnet56, cifar10_shufflenetv2_x2_0,
    cifar10_vgg16_bn, cifar10_vgg19_bn, cifar10_vit_b16, cifar10_repvgg_a0,
    cifar100_mobilenetv2_x0_5, cifar100_resnet20, cifar100_resnet56, cifar100_shufflenetv2_x2_0,
    cifar100_vgg16_bn, cifar100_vgg19_bn, cifar100_vit_b16, cifar100_repvgg_a0
)
from models.pytorch_cifar_models import DenseNet121 as cifar10_densenet121
from models.pytorch_cifar_models import EfficientNetB0 as cifar10_efficientnetb0
# from robustbench.utils import load_model as load_model_aa


BASE_PATH = "/home/mrliu/work/TEST_CODE/pytorch-cifar/checkpoint/"


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def load_clean_model(args, device=None):
    """
    Loads a pre-trained classification model for the specified dataset and architecture,
    and wraps it with input normalization.

    Args:
        args: argparse.Namespace with attributes: dataset, model
        device: torch.device

    Returns:
        A PyTorch nn.Module model (in eval mode, on the specified device)
    """

    # ====== CIFAR-10 Models ====== #
    if args.dataset == "CIFAR-10":
        if args.model == 'vgg16_bn':
            net = cifar10_vgg16_bn(pretrained=True)
        elif args.model == 'vgg19_bn':
            net = cifar10_vgg19_bn(pretrained=True)
        elif args.model == "resnet56":
            net = cifar10_resnet56(pretrained=True)
        elif args.model == "resnet20":
            net = cifar10_resnet20(pretrained=True)
        elif args.model == "mobilenetv2":
            net = cifar10_mobilenetv2_x0_5(pretrained=True)
        elif args.model == "shufflenetv2":
            net = cifar10_shufflenetv2_x2_0(pretrained=True)
        elif args.model == "repvgg":
            net = cifar10_repvgg_a0(pretrained=True)
        elif args.model == "vit_b16":
            net = cifar10_vit_b16(pretrained=True)
        elif args.model == "densenet121":
            net = cifar10_densenet121()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/kpt-densenet121.pth')
            net.load_state_dict(checkpoint['net'])
        elif args.model == "efficientnetb0":
            net = cifar10_efficientnetb0()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/ckpt-efficientnetb0.pth')
            net.load_state_dict(checkpoint['net'])
        else:
            print("Not implemented!!!", args.model)
            sys.exit(1)

        # Standard CIFAR-10 normalization
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        net = nn.Sequential(normalize, net)

    # ====== CIFAR-100 Models ====== #
    elif args.dataset == "CIFAR-100":
        if args.model == 'vgg16_bn':
            net = cifar100_vgg16_bn(pretrained=True)
        elif args.model == 'vgg19_bn':
            net = cifar100_vgg19_bn(pretrained=True)
        elif args.model == "resnet56":
            net = cifar100_resnet56(pretrained=True)
        elif args.model == "resnet20":
            net = cifar100_resnet20(pretrained=True)
        elif args.model == "mobilenetv2":
            net = cifar100_mobilenetv2_x0_5(pretrained=True)
        elif args.model == "shufflenetv2":
            net = cifar100_shufflenetv2_x2_0(pretrained=True)
        elif args.model == "repvgg":
            net = cifar100_repvgg_a0(pretrained=True)
        elif args.model == "vit_b16":
            net = cifar100_vit_b16(pretrained=True)
        elif args.model == "densenet121":
            net = cifar10_densenet121()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/ckpt-densenet121-cifar100.pth')
            net.load_state_dict(checkpoint['net'])
        elif args.model == "efficientnetb0":
            net = cifar10_efficientnetb0()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/ckpt-efficientnetb0-cifar100.pth')
            net.load_state_dict(checkpoint['net'])
        else:
            print("Not implemented!!!", args.model)
            sys.exit(1)

        # Standard CIFAR-100 normalization
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        net = nn.Sequential(normalize, net)

    # ====== STL-10 Models ====== #
    elif args.dataset == "STL-10":
        # Import models dynamically based on name
        try:
            if args.model.startswith('vgg'):
                from models.stl10.vgg import vgg16_bn, vgg13_bn, vgg11_bn, vgg19_bn
                net = eval(args.model + '_bn')()
            elif args.model.startswith('densenet'):
                from models.stl10.densenet import densenet121, densenet161, densenet169, densenet201
                net = eval(args.model)()
            elif args.model == 'googlenet':
                from models.stl10.googlenet import googlenet
                net = googlenet()
            # ... (rest omitted for brevity, see your original code)
            else:
                print('The network name you have entered is not supported yet')
                sys.exit(1)
        except Exception as e:
            print(f"Error loading STL-10 model: {e}")
            sys.exit(1)
        # Load checkpoint for STL-10 models
        checkpoint = torch.load(F'checkpoints/normal/{args.dataset}/{args.model}.pth')
        net.load_state_dict(checkpoint)

    # ====== ImageNet & NIPS2017 (Standard TorchVision) ====== #
    elif args.dataset in ["ImageNet", "NIPS2017"]:
        if args.model == 'inceptionv3':
            net = models.inception_v3(pretrained=True)
        elif args.model == 'vgg16':
            net = models.vgg16(pretrained=True)
        elif args.model == 'vgg19':
            net = models.vgg19(pretrained=True)
        elif args.model == "resnet50":
            net = models.resnet50(pretrained=True)
        elif args.model == "resnet152":
            net = models.resnet152(pretrained=True)
        elif args.model == "densenet121":
            net = models.densenet121(pretrained=True)
        elif args.model == "wide_resnet50":
            net = models.wide_resnet50_2(pretrained=True)
        elif args.model == "shufflenetv2":
            net = models.shufflenet_v2_x0_5(pretrained=True)
        elif args.model == "mobilenetv2":
            net = models.mobilenet_v2(pretrained=True)
        else:
            print("Not implemented!!!", args.model)
            sys.exit(1)

        # Standard ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        net = nn.Sequential(normalize, net)

    else:
        print("Not implemented!!!")
        sys.exit(1)

    # Set model to eval mode and move to device
    net.eval().to(device)

    return net
