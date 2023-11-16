# from skimage.measure import compare_psnr as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import compare_ssim as ssim
import numpy as np
import os
import torch
from collections import OrderedDict

def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    # print(im1.shape)
    # print(im2.shape)
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    # s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    #          multichannel=isRGB)
    return s

def compute_ssim_as(im1, im2):
    # print(im1.shape)
    # print(im2.shape)
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB, channel_axis=2)
    # s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    #          multichannel=isRGB)
    return s


def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims

def crop_forward(x, model, shave=32):
    ACS_GRID_SIZE = 4
    b, c, h, w = x.size()

    h_half, w_half = h // 2, w // 2

    h_size, w_size = h_half + shave - (h_half + shave) % ACS_GRID_SIZE, w_half + shave - (w_half + shave) % ACS_GRID_SIZE

    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    outputlist = []

    with torch.no_grad():
        if h % ACS_GRID_SIZE == 0 and w % ACS_GRID_SIZE == 0:
                return model(x)

        input_batch = torch.cat(inputlist, dim=0)
        output_batch = model(input_batch)
        # print("Output batch" + str(output_batch.shape))
        outputlist.extend(output_batch.chunk(ACS_GRID_SIZE, dim=0))

        output = torch.zeros_like(x)

        output[:, :, 0:h_half, 0:w_half] \
            = outputlist[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)


def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def convert2np(tensor):
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(path):

    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit
