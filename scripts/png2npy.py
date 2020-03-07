import os
import argparse
import skimage.io as sio
import numpy as np

parser = argparse.ArgumentParser(description='Pre-processing .png images')
parser.add_argument('--pathFrom', default='',
                    help='directory of images to convert')
parser.add_argument('--pathTo', default='',
                    help='directory of images to save')
parser.add_argument('--split', default=True,
                    help='save individual images')
parser.add_argument('--select', default='',
                    help='select certain path')

args = parser.parse_args()

for (path, dirs, files) in os.walk(args.pathFrom):
    print(path)
    targetDir = os.path.join(args.pathTo, path[len(args.pathFrom) + 1:])
    if len(args.select) > 0 and path.find(args.select) == -1:
        continue

    if not os.path.exists(targetDir):
        os.mkdir(targetDir)

    if len(dirs) == 0:
        pack = {}
        n = 0
        for fileName in files:
            (idx, ext) = os.path.splitext(fileName)
            if ext == '.png':
                image = sio.imread(os.path.join(path, fileName))
                if args.split:
                    np.save(os.path.join(targetDir, idx + '.npy'), image)
                n += 1
                if n % 100 == 0:
                    print('Converted ' + str(n) + ' images.')
