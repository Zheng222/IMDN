#import time
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
parser.add_argument("--acs", type=int, default=2,
                    help="height/width of Adaptive Cropping Strategy grid")
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--is_y", action='store_true', default=False,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder
if filepath.split('/')[-2] == 'Set5':
    ext = '.bmp'
else:
    ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model = architecture.IMDN_AS()
model_dict = torch.load(opt.checkpoint)
#model_dict = utils.load_state_dict(opt.checkpoint)
# print(model_dict)
model.load_state_dict(model_dict, strict=True)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
i = 0
for imname in filelist:
    # s = time.time()
    im_gt = cv2.imread(imname)[:, :, [2, 1, 0]]
    if ext == '.png': im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1])[:, :, [2, 1, 0]]
    else: im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1].split('.')[0] + 'x2' + ext, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()
    # Some datasets use smaller LR images, so ensure they have the same dimensions as the GT images
    if im_l.shape != im_gt.shape: im_input = torch.nn.functional.interpolate(im_input, size=(im_gt.shape[0], im_gt.shape[1]))
    # e = time.time()
    # print(e-s)
    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    _, _, h, w = im_input.size()
    with torch.no_grad():
        start.record()
        #s1 = time.time()
        out = utils.crop_forward(im_input, model, acs_xy=opt.acs)
        #e1 = time.time()
        #print(e1-s1)
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
    ssim_list[i] = utils.compute_ssim_as(im_pre, im_label)

    output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1])

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, sr_img[:, :, [2, 1, 0]])
    i += 1

print("Mean PSNR: {}, SSIM: {}, Time: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
