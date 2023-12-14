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
    return s

def compute_ssim_as(im1, im2):
    # print(im1.shape)
    # print(im2.shape)
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB, channel_axis=2)
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

def crop_forward(x, model, shave=32, acs_xy = 2):
    ACS_GRID_SIZE = acs_xy**2
    
    b, c, h, w = x.size()

    # Uncomment this to allow base IMDN to be used
    with torch.no_grad():
        # Don't bother with ACS unless its actually needed
        if h % ACS_GRID_SIZE == 0 and w % ACS_GRID_SIZE == 0:
                return model(x)

    # The unpadded dimensions of a single patch in the ACS grid
    h_chunk, w_chunk = h // acs_xy, w // acs_xy

    # Patch size plus any buffer needed
    h_size, w_size = h_chunk + shave - (h_chunk + shave) % ACS_GRID_SIZE, w_chunk + shave - (w_chunk + shave) % ACS_GRID_SIZE

    """print("h"+str(h))
    print("w"+str(w))
    print("h_size"+str(h_size))
    print("w_size"+str(w_size))
    print("h_chunk"+str(h_chunk))
    print("w_chunk"+str(w_chunk))
    print("-------------")"""
    inputlist = [0]*ACS_GRID_SIZE
    
    for h_patch in range(acs_xy):
        for w_patch in range(acs_xy):
            patch = acs_xy*h_patch+w_patch
            h_lb, w_lb = h_patch*h_size, w_patch*w_size # Patch's lower bounds
            h_ub, w_ub = (h_patch+1)*h_size, (w_patch+1)*w_size # Patch's upper bounds
            # Correct for any imprecision in patch size
            if(h_ub > h): 
                h_offset = h_ub - h
                h_ub -= h_offset
                h_lb -= h_offset
            if(w_ub > w):
                w_offset = w_ub - w
                w_ub -= w_offset
                w_lb -= w_offset
            
            """print("h["+str(patch)+"]: ("+str(h_lb)+", "+str(h_ub)+")")
            print("w["+str(patch)+"]: ("+str(w_lb)+", "+str(w_ub)+")")"""
            inputlist[patch] = x[:, :, h_lb:h_ub, w_lb:w_ub]
            # print("patch["+str(patch)+"]: "+str(inputlist[patch].shape))

    outputlist = []

    with torch.no_grad():
        input_batch = torch.cat(inputlist, dim=0)
        output_batch = model(input_batch)
        # print("Output batch" + str(output_batch.shape))
        outputlist.extend(output_batch.chunk(ACS_GRID_SIZE, dim=0))

        output = torch.zeros_like(x)
        
    # Once SR is calculated, put each patch back where it's supposed to be and remove any excess padding
    for h_patch in range(acs_xy):
        for w_patch in range(acs_xy):
            patch = acs_xy*h_patch+w_patch
            h_lb_chunk, w_lb_chunk = h_patch*h_chunk, w_patch*w_chunk # Lower bounds of the "chunk", AKA the unpadded patch size
            h_ub_chunk, w_ub_chunk = (h_patch+1)*h_chunk, (w_patch+1)*w_chunk # Upper bounds of the "chunk", AKA the unpadded patch size
            
            # Correct for any imprecision in patch size
            if(h_ub_chunk > h): 
                h_offset = h_ub_chunk - h
                h_ub_chunk -= h_offset
                h_lb_chunk -= h_offset
            if(w_ub_chunk > w):
                w_offset = w_ub_chunk - w
                w_ub_chunk -= w_offset
                w_lb_chunk -= w_offset
                
            """print("h_chunk["+str(patch)+"]: ("+str(h_lb_chunk)+", "+str(h_ub_chunk)+")")
            print("w_chunk["+str(patch)+"]: ("+str(w_lb_chunk)+", "+str(w_ub_chunk)+")")"""
            
            h_lb, w_lb = h_patch*h_size, w_patch*w_size # Lower bounds of the original patch's dimensions
            h_ub, w_ub = (h_patch+1)*h_size, (w_patch+1)*w_size # Upper bounds of the original patch's dimensions
            
            # Correct for any imprecision in patch size
            if(h_ub > h): 
                h_offset = h_ub - h
                h_ub -= h_offset
                h_lb -= h_offset
            if(w_ub > w):
                w_offset = w_ub - w
                w_ub -= w_offset
                w_lb -= w_offset

            """print("h["+str(patch)+"]: ("+str(h_lb)+", "+str(h_ub)+")")
            print("w["+str(patch)+"]: ("+str(w_lb)+", "+str(w_ub)+")")"""

            # Lower bounds of the patch that should be taken from the output
            h_out_lb = np.int32(np.rint((h_patch*h_size-h_lb_chunk)/(acs_xy-1)))
            w_out_lb = np.int32(np.rint((w_patch*w_size-w_lb_chunk)/(acs_xy-1)))
            # Upper bounds of the patch that should be taken from the output
            h_out_ub = h_ub_chunk - h_lb + (h_patch+acs_xy-1) * h_out_lb
            w_out_ub = w_ub_chunk - w_lb + (w_patch+acs_xy-1) * w_out_lb

            # Correct for any imprecision in patch size
            if(h_out_ub-h_out_lb != h_ub-h_lb): 
                h_offset = (h_out_ub - h_out_lb) - (h_ub - h_lb)
                h_out_ub = h_size
                h_out_lb = h_size - (h_ub - h_lb)
            if(w_out_ub-w_out_lb != w_ub-w_lb):
                w_offset = (w_out_ub - w_out_lb) - (w_ub - w_lb)
                w_out_ub = w_size
                w_out_lb = w_size - (w_ub - w_lb)

            """print("h_out["+str(patch)+"]: ("+str(h_out_lb)+", "+str(h_out_ub)+")")
            print("w_out["+str(patch)+"]: ("+str(w_out_lb)+", "+str(w_out_ub)+")")"""
            
            # Place the HR patch back where it was originally in the LR image
            output[:, :, h_lb:h_ub, w_lb:w_ub] = outputlist[patch][:, :, h_out_lb:h_out_ub, w_out_lb:w_out_ub]

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
