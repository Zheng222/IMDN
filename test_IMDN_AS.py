import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import architecture

parser = argparse.ArgumentParser(description='IMDN_AS')
parser.add_argument("--test_hr_folder", type=str, default='Test_Datasets/RealSR/ValidationGT',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='Test_Datasets/RealSR/ValidationLR/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/RealSR')
parser.add_argument("--checkpoint", type=str, default='checkpoints/IMDN_AS.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--is_y", action='store_true', default=False,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')


def crop_forward(x, model, shave=32):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2

    h_size, w_size = h_half + shave - (h_half + shave) % 4, w_half + shave - (w_half + shave) % 4

    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    outputlist = []

    with torch.no_grad():
        input_batch = torch.cat(inputlist, dim=0)
        output_batch = model(input_batch)
        outputlist.extend(output_batch.chunk(4, dim=0))

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


filepath = opt.test_hr_folder
if filepath.split('/')[-2] == 'Set5' or filepath.split('/')[-2] == 'Set14':
    ext = '.bmp'
else:
    ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model = architecture.IMDN_AS()
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
i = 0
for imname in filelist:
    im_gt = cv2.imread(imname)[:, :, [2, 1, 0]]
    im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1])[:, :, [2, 1, 0]]
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    _, _, h, w = im_input.size()
    with torch.no_grad():

        if h % 4 == 0 and w % 4 == 0:
            start.record()
            out = model(im_input)
            end.record()
            torch.cuda.synchronize()
            time_list[i] = start.elapsed_time(end)  # milliseconds
        else:
            start.record()
            out = crop_forward(im_input, model)
            end.record()
            torch.cuda.synchronize()
            time_list[i] = start.elapsed_time(end)  # milliseconds

    sr_img = utils.tensor2np(out.detach()[0])
    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(im_gt)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(sr_img)[:, :, 0])
    else:
        im_label = im_gt
        im_pre = sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)

    output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1])

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, sr_img[:, :, [2, 1, 0]])
    i += 1

print("Mean PSNR: {}, SSIM: {}, Time: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
