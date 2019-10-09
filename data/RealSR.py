import os
from data import common
import numpy as np
import torch.utils.data as data
from data.image_folder import make_dataset


class srdata(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.root = self.args.root
        self.split = 'train' if train else 'test'
        self.scale = 1
        self.repeat = args.test_every // (args.n_train // args.batch_size)  # 300 / (60/6) = 30
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan_npy()

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/RealSR_decoded'
        self.dir_hr = os.path.join(self.apath, 'TrainGT')
        self.dir_lr = os.path.join(self.apath, 'TrainLR')
        self.ext = '.npy'

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return self.args.n_train * self.repeat
        else:
            return self.args.n_val

    def _get_index(self, idx):
        if self.train:
            return idx % self.args.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.args.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)

        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]

        return img_in, img_tar

    def _scan(self):
        list_hr = []
        list_lr = []
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>6}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            list_lr.append(os.path.join(self.dir_lr, '{}x{}{}'.format(filename, self.scale, self.ext)))

        return list_hr, list_lr

    def _scan_npy(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = np.load(self.images_lr[idx])
        hr = np.load(self.images_hr[idx])
        return lr, hr
