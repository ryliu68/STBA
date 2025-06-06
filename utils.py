import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from IQA_pytorch import DISTS, SSIM, LPIPSvgg # 24.03
import pyiqa  # For perceptual similarity metrics | 2025.06
from einops import rearrange, reduce, repeat
import torchgeometry

warnings.filterwarnings("ignore")

# 2024.03
# lpips_fn = LPIPSvgg().cuda()
# ssim_fn = SSIM().cuda()
# dists_fn = DISTS().cuda()


# 2025.06
# Initialize perceptual similarity metric functions (using GPU if available)
lpips_fn = pyiqa.create_metric('lpips', device="cuda:0")
ssim_fn = pyiqa.create_metric('ssim', device="cuda:0")
dists_fn = pyiqa.create_metric('dists', device="cuda:0")


def distance_2(x_adv, x, norm='l2'):
    """
    Compute the distance between two images, supporting various norms and perceptual metrics.
    Args:
        x_adv: Tensor, adversarial image
        x: Tensor, original image
        norm: 'l2' | 'linf' | 'lpips' | 'dists' | 'ssim'
    Returns:
        out: float, the computed distance/metric value
    """
    if norm == 'l2':
        out = torch.norm(x_adv - x, p=2).item()
    elif norm == 'linf':
        out = torch.max(torch.abs(x_adv - x)).item()
    elif norm == 'lpips':
        out = lpips_fn(x_adv, x).item()
    elif norm == 'dists':
        out = dists_fn(x_adv, x).item()
    elif norm == 'ssim':
        out = ssim_fn(x_adv, x).item()
    else:
        raise ValueError("Unsupported norm type.")
    return out


def flow_st(images, flows):
    """
    Apply a dense flow field (optical flow-like) spatial transformation to a batch of images,
    using bilinear interpolation.
    Args:
        images: (B, C, H, W) input tensor
        flows: (B, 2, H, W) flow tensor, where 2 = [y, x] shift for each pixel
    Returns:
        perturbed_image: (B, C, H, W) the flow-warped image
    """
    batch_size, _, H, W = images.size()
    device = images.device

    # Create mesh grid for pixel locations
    grid_single = torch.stack(
        torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    ).float()
    grid = grid_single.repeat(batch_size, 1, 1, 1).to(device)

    images = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    grid_new = grid + flows

    sampling_grid_x = torch.clamp(grid_new[:, 1], 0., W - 1.)
    sampling_grid_y = torch.clamp(grid_new[:, 0], 0., H - 1.)

    # Nearest neighbors for bilinear interpolation
    x0 = torch.floor(sampling_grid_x).long()
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)

    b = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, H, W).to(device)
    Ia = images[b, y0, x0].float()
    Ib = images[b, y1, x0].float()
    Ic = images[b, y0, x1].float()
    Id = images[b, y1, x1].float()

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()
    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)

    perturbed_image = wa * Ia + wb * Ib + wc * Ic + wd * Id
    perturbed_image = perturbed_image.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
    return perturbed_image


class Flow(nn.Module):
    """
    A learnable spatial flow field module for adversarial transformation.
    """

    def __init__(self, height, width, parameterization=None):
        super().__init__()
        self.H = height
        self.W = width
        self.parameterization = parameterization if parameterization else nn.Identity()
        self._pre_flow_field = nn.Parameter(
            torch.randn([2, self.H, self.W]) * 0.1, requires_grad=True
        )

    def forward(self, x):
        assert (self.H == x.shape[-2] and self.W == x.shape[-1]), \
            "flow is initialized with different shape than image"
        BATCH_SIZE = x.shape[0]
        applied_flow_field = self.parameterization(self._pre_flow_field)

        grid = torch.stack(torch.meshgrid(torch.arange(self.H), torch.arange(self.W))).to(self._pre_flow_field.device)
        batched_grid = repeat(grid, "c h w -> b h w c", b=BATCH_SIZE)
        applied_flow_field_batch = repeat(applied_flow_field, "yx h w -> b h w yx", b=BATCH_SIZE)
        sampling_grid = batched_grid + applied_flow_field_batch
        return self.sample_grid(x, sampling_grid)

    def sample_grid(self, img_batch, grid_batch):
        """
        Bilinear interpolation for sampling grid locations.
        """
        num_channels = img_batch.shape[1]
        added = repeat(
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=img_batch.device),
            "ind c -> b h w ind c",
            b=img_batch.shape[0], h=img_batch.shape[-2], w=img_batch.shape[-1]
        )
        sampled_pixel_coordinates = torch.add(
            repeat(torch.floor(grid_batch).long(), "b h w c -> b h w 4 c"), added
        )
        sampled_pixel_distances = torch.abs(
            sampled_pixel_coordinates - repeat(grid_batch, "b h w c -> b h w c2 c", c2=4)
        )
        sampled_pixel_distances_h = sampled_pixel_distances[:, :, :, :, 0]
        sampled_pixel_distances_w = sampled_pixel_distances[:, :, :, :, 1]
        sampled_pixel_weights = (1 - sampled_pixel_distances_h) * (1 - sampled_pixel_distances_w)

        sampled_pixel_coordinates_y = torch.clamp(sampled_pixel_coordinates[:, :, :, :, 0], 0, self.H - 1)
        sampled_pixel_coordinates_x = torch.clamp(sampled_pixel_coordinates[:, :, :, :, 1], 0, self.W - 1)
        sampled_pixel_indices = (sampled_pixel_coordinates_y * self.W + sampled_pixel_coordinates_x)

        sampled_pixel_indices_flat = repeat(
            sampled_pixel_indices, "b h w four -> b c (h w four)", c=num_channels
        )
        img_batch_flat = rearrange(img_batch, "b c h w -> b c (h w)")
        sampled_pixels_flat = torch.gather(img_batch_flat, -1, sampled_pixel_indices_flat)
        sampled_pixels = rearrange(
            sampled_pixels_flat, "b c (h w four) -> b c h w four", h=self.H, w=self.W, four=4
        )
        sampled_pixels_weighted = sampled_pixels * repeat(
            sampled_pixel_weights, "b h w four -> b c h w four", c=num_channels
        )
        sampled_pixels_weighted_sum = reduce(
            sampled_pixels_weighted, "b c h w four -> b c h w", reduction="sum"
        )
        return sampled_pixels_weighted_sum

    def get_applied_flow(self):
        return self.parameterization(self._pre_flow_field)


def cw_loss(args, logits, labels, targeted=False):
    """
    Carlini-Wagner (CW) loss function for adversarial attacks.
    Args:
        args: arguments with num_classes
        logits: prediction logits (tensor, [B, num_classes])
        labels: ground truth labels (tensor, [B])
    Returns:
        loss: tensor, [B]
    """
    target_onehot = torch.eye(args.num_classes, device=labels.device)[labels]
    real = torch.log((target_onehot * logits).sum(1) + 1e-10)
    other = torch.log(
        ((1. - target_onehot) * logits - target_onehot * 10000.).max(1)[0] + 1e-10)
    loss = torch.clamp(real - other, 0., 1000.)
    return loss


class Loss_flow(nn.Module):
    """
    Flow smoothness regularization: penalizes large spatial changes in the flow field.
    """

    def __init__(self, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
        super().__init__()

    def forward(self, flows):
        paddings = (1, 1, 1, 1, 0, 0, 0, 0)
        padded_flows = F.pad(flows, paddings, "constant", 0)
        shifted_flowsr = torch.stack([
            padded_flows[:, :, 2:, 1:-1],  # bottom mid
            padded_flows[:, :, 1:-1, :-2],  # mid left
            padded_flows[:, :, :-2, 1:-1],   # top mid
            padded_flows[:, :, 1:-1, 2:],  # mid right
        ], -1)
        flowsr = flows.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        _, h, w, _ = flowsr[:, 0].shape
        loss0 = torch.norm((flowsr[:, 0] - shifted_flowsr[:, 0]).view(-1, 4), p=2, dim=0, keepdim=True) ** 2
        loss1 = torch.norm((flowsr[:, 1] - shifted_flowsr[:, 1]).view(-1, 4), p=2, dim=0, keepdim=True) ** 2
        return torch.max(torch.sqrt((loss0 + loss1) / (h * w)))


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_data_and_model(args):
    """
    Set dataset-specific model parameters and file paths for batch data.
    Also sets the list of model names for attack evaluation.
    """
    if args.dataset == "CIFAR-10":
        args.image_size = 32
        args.img_dims = [3, 32, 32]
        args.num_classes = 10
        args.k = 9
        args.field_size = [1, 2, 32, 32]
        if args.target_model == "defense":
            args.batchfile = F"data/CIFAR-10_Sehwag2020Hydra_Wang2020Improving_Zhang2019Theoretically_Wong2020Fast_robust.pth"
            args.model_names = ["Sehwag2020Hydra", "Wang2020Improving", "Zhang2019Theoretically", "Wong2020Fast"]
        else:
            args.batchfile = F"data/CIFAR-10_vgg19_bn_resnet56_mobilenetv2_shufflenetv2.pth"
            args.model_names = ["vgg19_bn", "resnet56", "mobilenetv2", "shufflenetv2"]
    elif args.dataset == "CIFAR-100":
        args.image_size = 32
        args.img_dims = [3, 32, 32]
        args.num_classes = 100
        args.k = 9
        args.field_size = [1, 2, 32, 32]
        if args.target_model == "defense":
            args.batchfile = F"data/CIFAR-100_Pang2022Robustness_Addepalli2022Efficient_Sehwag2021Proxy_Cui2020Learnable_robust.pth"
            args.model_names = ["Pang2022Robustness", "Addepalli2022Efficient", "Sehwag2021Proxy", "Cui2020Learnable"]
        else:
            args.batchfile = F"data/CIFAR-100_vgg19_bn_resnet56_mobilenetv2_shufflenetv2.pth"
            args.model_names = ["vgg19_bn", "resnet56", "mobilenetv2", "shufflenetv2"]
    elif args.dataset == "STL-10":
        args.image_size = 96
        args.img_dims = [3, 96, 96]
        args.num_classes = 10
        args.k = 64
        args.field_size = [1, 2, 96, 96]
        if args.target_model == "defense":
            args.batchfile = F"data/STL-10_FREE_FAST_TRADES_MART.pth"
            args.model_names = ["FREE", "FAST", "TRADES", "MART"]
        else:
            args.batchfile = F"data/STL-10_vgg19_bn_resnet56_mobilenetv2_shufflenetv2.pth"
            args.model_names = ["vgg19_bn", "resnet56", "mobilenetv2", "shufflenetv2"]
    elif args.dataset == "ImageNet":
        args.image_size = 224
        args.img_dims = [3, 224, 224]
        args.num_classes = 1000
        args.k = 40
        args.field_size = [1, 2, 224, 224]
        if args.target_model == "defense":
            args.batchfile = "data/ImageNet_Salman2020Do_50_2_Salman2020Do_R50_Engstrom2019Robustness_Wong2020Fast_Salman2020Do_R18_Standard_R50_robust.pth"
            args.model_names = ["Salman2020Do_50_2", "Salman2020Do_R50", "Engstrom2019Robustness", "Wong2020Fast", "Salman2020Do_R18"]
        else:
            args.batchfile = "data/ImageNet_vgg19_resnet152_mobilenetv2_shufflenetv2.pth"
            args.model_names = ["vgg19", "resnet152", "mobilenetv2", "densenet121"]
    else:
        print("Not implemented yet")


def calc_Freq(args, torch_img, kernel=3):
    """
    Decompose an image into low-frequency and high-frequency components
    using Gaussian blur. Used for frequency-based attacks or analysis.
    Args:
        args: Argument object with dataset type
        torch_img: input tensor (B, C, H, W)
        kernel: size of Gaussian kernel
    Returns:
        new_lowFreq: low frequency component (masked)
        new_highFreq: high frequency component (masked)
    """
    if kernel == 3:
        sigma = 3
    elif kernel == 5:
        sigma = 1.5
    else:
        sigma = 1
    lowFreq = torchgeometry.image.gaussian_blur(
        torch_img, (kernel, kernel), (sigma, sigma)
    )
    highFreq = torch_img - lowFreq

    if args.dataset == "ImageNet":
        mask = torch.zeros([1, 1, 224, 224]).to(torch_img.device)
        mask[:, :, 3:222, 3:222] = 1
    elif args.dataset == "STL-10":
        mask = torch.zeros([1, 1, 96, 96]).to(torch_img.device)
        mask[:, :, 3:94, 3:94] = 1
    else:
        mask = torch.zeros([1, 1, 32, 32]).to(torch_img.device)
        mask[:, :, 3:30, 3:30] = 1

    new_highFreq = mask * highFreq
    new_lowFreq = lowFreq + (1 - mask) * highFreq
    return new_lowFreq, new_highFreq
