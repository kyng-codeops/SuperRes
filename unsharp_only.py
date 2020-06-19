#!/usr/bin/env python3

import cv2
from scipy.ndimage.filters import median_filter
import argparse
import sys
import os
import numpy as np
import time
import multiprocessing

imw_code = {
    'jpg': ([cv2.IMWRITE_JPEG_QUALITY, 87], 'jpg'),
    'j90': ([cv2.IMWRITE_JPEG_QUALITY, 90], 'jpg'),
    'j100': ([cv2.IMWRITE_JPEG_QUALITY, 100], 'jpg'),
    'png': ([cv2.IMWRITE_PNG_COMPRESSION, 1], 'png'),
    'png3': ([cv2.IMWRITE_PNG_COMPRESSION, 4], 'png'),
    'png9': ([cv2.IMWRITE_PNG_COMPRESSION, 9], 'png'),
    'wpll': ([cv2.IMWRITE_WEBP_QUALITY, 101], 'webp')
}
I_O_EXT = 'png9'

def sharpen_channel(image, sigma, strength):
    t0 = time.time()

    mf_image = median_filter(image, sigma)
    lap = cv2.Laplacian(mf_image, cv2.CV_64F)
    result = image - strength * lap
    result[result > 255] = 255
    result[result < 0] = 0

    t1 = time.time()
    dt = t1 - t0
    return result, dt

def sharpen_rgb_bgr(image, sigma, strength):
    r1 = np.zeros_like(image)
    dt2 = 0
    for i in range(3):
        # r1[:, :, i], dt = sharpen(result[:, :, i], 1, 0.2)  # max strength if before upscale
        r1[:, :, i], dt = sharpen_channel(image[:, :, i], sigma, strength)
        dt2 = dt2 + dt
    
    result = r1
    return result, dt2

def process_pipeline(f, sigma, strength, o_dir, o_ext):
    image = cv2.imread(f)
    result, dt = sharpen_rgb_bgr(image, sigma, strength)

    workdir = os.getcwd()
    o_path = '{}/{}'.format(workdir, o_dir)
    f_label = f.split('.')[:-1][0]
    o_fqn = '{}/{}.{}'.format(o_path, f_label, imw_code[o_ext][1])
    was_wrtn = cv2.imwrite(o_fqn, result, imw_code[o_ext][0])

    if was_wrtn:
        unsharp_out = 'unsharp {:.{prec}f} sec'.format(dt, prec=2)
        print('output\t{}:\t{}'.format(f, unsharp_out))
        return o_fqn
    return False

def get_cli_args(args):
    parser = argparse.ArgumentParser(description='unsharp images')
    parser.add_argument('-s', '--sigma', nargs=1, required=True,
        help='integer for sigma of sharpening')
    parser.add_argument('-m', '--strength', nargs=1, required=True,
        help='float for strength of sharpening .7 is very light 10 too much')
    parser.add_argument('-o', '--out', nargs=1, required=False, default=I_O_EXT,
        choices=imw_code.keys(),
        help='string tag to define output type')
    subparsers = parser.add_subparsers(help='provide files or imseq')

    f_parser = subparsers.add_parser('files')
    i_parser = subparsers.add_parser('imseq')

    f_parser.add_argument('file', nargs='*',
        help='file_name or file_name_pattern')
    
    i_parser.add_argument('-b', '--begin', nargs=1, help='starting img 0001.png')
    i_parser.add_argument('-e', '--end', nargs=1, help='ending img 0010.png')
    return parser.parse_args(args)

def main(args):
    dt_set = get_cli_args(args)    
    
    sigma = int(dt_set.sigma[0])
    strength = float(dt_set.strength[0])
    
    if isinstance(dt_set.out, list):
        o_ext = dt_set.out[0]
    else:
        o_ext = dt_set.out
    
    im_files_flg = False
    
    try:
        file_pattern = dt_set.file
        im_files_flg = True
    except AttributeError:
        seq_b = dt_set.begin[0]
        seq_e = dt_set.end[0]
        if len(seq_b) != len(seq_e):
            raise RuntimeError('Length of chars in begining and ending inconsistant')
    
    # Create a subfolder for output using model label name with x factor
    o_dir = 'Unsharp_out'
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)

    o_list = []
    batch_args = []

    if im_files_flg:
        for f in file_pattern:
            batch_args.append((f, sigma, strength, o_dir, o_ext))

    else:
        f = seq_b.split('.')
        if os.path.isfile('{}.{}'.format(f[0], f[1])):
            ext = f[1]
            pad = len(f[0])
            frame_b = int(f[0])
            frame_e = int(seq_e.split('.')[0]) + 1
            
            for frame in range(frame_b, frame_e, 1):
                f_name = '{:0>{width}}.{}'.format(frame, ext, width=pad)
                batch_args.append((f_name, sigma, strength, o_dir, o_ext))
            
    n_proc = min(multiprocessing.cpu_count(), len(batch_args))
    
    pool = multiprocessing.Pool(n_proc)
    o_list = pool.starmap(process_pipeline, batch_args)
    pool.close()
    pool.join()
    
    # reorder all mtime atime from out-of-order processing
    for out_f in o_list:
        os.utime(out_f)

    return o_list


if __name__ == "__main__":
    main(sys.argv[1:])