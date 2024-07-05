# deep image prior type idea (or just learn color matrices directly)

# 6 images being produced with learned colors and alpha values
# weight decay or something similar to promote smoother output

# after color decompositions are found, recombine

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.dip import DIP


DEVICE = torch.device("mps")


def show(image, title=""):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def total_variation_loss(img):
    """
    docstering
    """
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


class Layer(nn.Module):
    def __init__(self, size):
        super().__init__()

        dip_args = {
            "z_channels": 3,
            "z_scale": 1 / 10,
            "z_noise": 0,
            "filters_down": [8, 16, 32, 64],
            "filters_up": [8, 16, 32, 64],
            "kernels_down": [3, 3, 3, 3],
            "kernels_up": [3, 3, 3, 3],
            "filters_skip": [3, 3, 3, 3],
            "kernels_skip": [1, 1, 1, 1],
            "upsampling": "nearest",
        }

        self.mask = DIP(size[0], 1, **dip_args).to(DEVICE)

        self.color = nn.Parameter(torch.randn(1, 1, 3, device=DEVICE))

    # output, loss
    def forward(self):
        mask = self.mask().squeeze(0).permute(1, 2, 0)
        color = F.sigmoid(self.color)

        return mask, color

    def show(self, round_mask=True):
        mask = self.mask().squeeze(0).permute(1, 2, 0)
        if round_mask:
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
        color = F.sigmoid(self.color)

        return show((mask * color + 1 - mask).detach().cpu().numpy())


class Reconstruction(nn.Module):
    def __init__(self, layers, size):
        super().__init__()
        self.size = size
        self.layers = nn.ParameterList([Layer(size) for _ in range(layers)])
        self.alpha = 0.4

    def forward(self, save_layers=False):
        saved_layers = []

        im = torch.ones(*self.size[:2], 3, device=DEVICE)

        for l in self.layers:
            mask, color = l()
            alpha_mask = mask * self.alpha
            im = im * (1 - alpha_mask) + color * alpha_mask
            if save_layers:
                saved_layers.append((mask, color))

        return im, saved_layers

    def show(self, round_mask=True):
        im = torch.ones(*self.size[:2], 3, device=DEVICE)

        for l in self.layers:
            mask, color = l()
            if round_mask:
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
            alpha_mask = mask * self.alpha
            im = im * (1 - alpha_mask) + color * alpha_mask

        show(im.detach().cpu().numpy())


if __name__ == "__main__":
    # load image
    im = cv.imread("images/example.png")

    im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32) / 255
    height, width, _ = im.shape

    new_width = 512
    new_height = 512
    # im = cv.resize(im, (new_width, new_width * height // width))
    im = cv.resize(im, (new_width, new_height))

    # im = cv.GaussianBlur(im, (0,0), 3)
    # show(im)

    reconst = Reconstruction(12, im.shape)
    optimizer = torch.optim.Adam(reconst.parameters(), lr=0.01)

    target = torch.tensor(im, device=DEVICE)

    it = tqdm.tqdm(range(500))
    for epoch in it:
        mask_extreme = 0

        y, layers = reconst(save_layers=True)

        loss = torch.mean(torch.norm(y - target, dim=2))
        for mask, color in layers:
            # loss = loss + total_variation_loss(mask.permute(2,0,1).unsqueeze(0))*10
            loss = loss + 1 / torch.mean(torch.abs(mask - 0.5)) * 0.02

        it.set_description(f"(Loss: {loss.item():.4f})")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for l in reconst.layers:
        l.show()
    reconst.show()
