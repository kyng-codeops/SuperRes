#!/usr/bin/env python3

import cv2
from scipy.ndimage.filters import median_filter
import argparse
import sys
import os
import numpy as np
import time
import multiprocessing

IMW_CODE = {
    'jpg': ([cv2.IMWRITE_JPEG_QUALITY, 87], 'jpg'),
    'j90': ([cv2.IMWRITE_JPEG_QUALITY, 90], 'jpg'),
    'j100': ([cv2.IMWRITE_JPEG_QUALITY, 100], 'jpg'),
    'png': ([cv2.IMWRITE_PNG_COMPRESSION, 1], 'png'),
    'png3': ([cv2.IMWRITE_PNG_COMPRESSION, 4], 'png'),
    'png9': ([cv2.IMWRITE_PNG_COMPRESSION, 9], 'png'),
    'wpll': ([cv2.IMWRITE_WEBP_QUALITY, 101], 'webp')
}
I_O_EXT = 'png9'
I_O_DIR = 'Denoise_out'

def denoise_rgb_bgr(image, kv_args):
    t0 = time.time()
    hL = kv_args['h_luma']
    hC = kv_args['h_croma']
    # b0, g0, r0 = cv2.split(image)
    dst = cv2.fastNlMeansDenoisingColored(image, None, hL, hC, 7, 21)
    # b1, g1, r1 = cv2.split(dst)
    # dst = cv2.merge([b1, g1, r0])
    dt = time.time() - t0
    return dst, dt 

def process_pipeline(f, kv_args):
    image = cv2.imread(f)
    result, dt = denoise_rgb_bgr(image, kv_args)

    workdir = os.getcwd()
    o_path = '{}/{}'.format(workdir, kv_args['o_dir'])
    f_label = f.split('.')[:-1][0]
    o_fqn = '{}/{}.{}'.format(o_path, f_label, IMW_CODE[kv_args['o_ext']][1])
    was_wrtn = cv2.imwrite(o_fqn, result, IMW_CODE[kv_args['o_ext']][0])

    if was_wrtn:
        pl_out = 'denoise {:.{prec}f} sec'.format(dt, prec=2)
        print('output\t{}:\t{}'.format(f, pl_out))
        return o_fqn
    return False

def ammend_ns(ns):
    # Custom kv args w defaults (for super class overloading)
    ns.kw_cus_arg.update({'h_luma': int(ns.h_luma)})
    ns.kw_cus_arg.update({'h_croma': int(ns.h_croma)})
    return ns

def preprocess_cli_ns(parser, args):
    arg_pkg = parser.parse_args(args)

    # flatten namespace single item lists where nargs=1
    for k, v in arg_pkg.__dict__.items():
        if isinstance(v, list):
            if len(v) == 1:
                arg_pkg.__dict__.update({k: v[0]})

    # For args with defaults to friendly kv
    arg_pkg.kw_cus_arg = {'o_ext': arg_pkg.out}
    arg_pkg.kw_cus_arg.update({'o_dir': arg_pkg.out_dir})

    # Custom kv args w defaults (for super class overloading)
    # arg_pkg.kw_cus_arg.update({'h_luma': int(arg_pkg.h_luma)})
    # arg_pkg.kw_cus_arg.update({'h_croma': int(arg_pkg.h_croma)})

    return arg_pkg

def ammend_clui(parser):
    parser.add_argument('-hL', '--h-luma', nargs=1, required=False, default=5,
        help='integer for h on luma channel')
    parser.add_argument('-hC', '--h-croma', nargs=1, required=False, default=5,
        help='integer for h_croma of color denoise')

    return parser

def reusable_default_clui():
    parser = argparse.ArgumentParser(description='Denoise image files')
    parser.add_argument('-o', '--out', nargs=1, required=False, default=I_O_EXT,
        choices=IMW_CODE.keys(),
        help='string tag to define output type')
    parser.add_argument('-d', '--out-dir', nargs=1, required=False, default=I_O_DIR,
        help='override the default output subfolder name')
    
    subparsers = parser.add_subparsers(help='provide files or image sequence numbers')

    f_parser = subparsers.add_parser('files')
    i_parser = subparsers.add_parser('imseq')

    f_parser.add_argument('file', nargs='*',
        help='file_name or file_name_pattern')
    
    i_parser.add_argument('-b', '--begin', nargs=1, help='example start with 0001.png')
    i_parser.add_argument('-e', '--end', nargs=1, help='example end with 0010.png')
    
    return parser
    
def main(args):
    # cli_set = get_cli_args(args)  
    parser = reusable_default_clui()
    parser = ammend_clui(parser)
    ns = preprocess_cli_ns(parser, args)
    ap_set = ammend_ns(ns)

    im_files_flg = False
    
    try:
        file_pattern = ap_set.file
        im_files_flg = True
    except AttributeError:
        seq_b = ap_set.begin
        seq_e = ap_set.end
        if len(seq_b) != len(seq_e):
            raise RuntimeError('Length of chars in begining and ending inconsistant')
    
    # Create a subfolder for output using model label name with x factor
    if not os.path.isdir(ap_set.kw_cus_arg['o_dir']):
        os.mkdir(ap_set.kw_cus_arg['o_dir'])

    o_list = []
    batch_args = []

    if im_files_flg:
        for f in file_pattern:
            batch_args.append(
                (f, ap_set.kw_cus_arg)
                )
    else:
        f = seq_b.split('.')
        if os.path.isfile('{}.{}'.format(f[0], f[1])):
            ext = f[1]
            pad = len(f[0])
            frame_b = int(f[0])
            frame_e = int(seq_e.split('.')[0]) + 1
            
            for frame in range(frame_b, frame_e, 1):
                f_name = '{:0>{width}}.{}'.format(frame, ext, width=pad)
                batch_args.append(
                    (f_name, ap_set.kw_cus_arg)
                    )            
    # # debugging multiprocess in vscode seems to still be a problem
    # for b_arg in batch_args:
    #     f, kv_args = [tups for tups in b_arg]
    #     ofl = process_pipeline(f, kv_args)
    #     o_list.append(ofl)

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