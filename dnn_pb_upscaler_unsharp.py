#!/usr/bin/env python3

import cv2
from cv2 import dnn_superres
from scipy.ndimage.filters import median_filter
import numpy as np
import plaidml.keras
# import imutils
import time
import os
import sys
import glob
import argparse

cv2.ocl.setUseOpenCL(True)

MOD_NAME = 'espcn'
UP_SIZE = 4
HOMEDIR = os.getenv('HOME')
PRE_BUILT = {
    'edsr': 'EDSR',
    'espcn': 'ESPCN',
    'lapsrn': 'LapSRN',
    'fsrcnn': 'FSRCNN',
    'fsrcnn-sm': 'FSRCNN-small'
}
M_FNAME = '{}_x{}.pb'.format(PRE_BUILT[MOD_NAME], UP_SIZE)
M_FQP = os.path.dirname(os.path.realpath(__file__))
M_FQN = '{}/{}'.format(M_FQP, M_FNAME)
M_LABEL = M_FQN.split('/')[-1].split('.')[0]

# j100 is fast but still using lossly schema (noise can propagate thru pipelines)
# png 0-9, default is 1 (fastest), 3 tested fair trade-off speed/size
# WebP 0-100, >100 triggers lossless (and as slow as png set to 9)
imw_code = {
    'jpg': ([cv2.IMWRITE_JPEG_QUALITY, 87], 'jpg'),
    'j90': ([cv2.IMWRITE_JPEG_QUALITY, 90], 'jpg'),
    'j100': ([cv2.IMWRITE_JPEG_QUALITY, 100], 'jpg'),
    'png': ([cv2.IMWRITE_PNG_COMPRESSION, 1], 'png'),
    'png3': ([cv2.IMWRITE_PNG_COMPRESSION, 4], 'png'),
    'png9': ([cv2.IMWRITE_PNG_COMPRESSION, 9], 'png'),
    'wpll': ([cv2.IMWRITE_WEBP_QUALITY, 101], 'webp')
}
V_O_EXT = 'png9'
I_O_EXT = 'jpg'

# Create a global SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Handy debugging
# before = img.shape()
# after = result.shape()

def set_sr_model(model, scale):
    # TODO: currently unable to compile opencv for CUDA on macOS 10.15.x
    # CUDA driver and toolkit support problems on macOS mojave
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    m_fname = '{}_x{}.pb'.format(PRE_BUILT[model], scale)
    m_fqp = os.path.dirname(os.path.realpath(__file__))
    m_fqn = '{}/{}'.format(m_fqp, m_fname)
    # M_LABEL = M_FQN.split('/')[-1].split('.')[0]
    sr.readModel(m_fqn)
    sr.setModel(model, scale)
    # TODO: FSRCNN has been ported to tf, will need to port tf to keras for GPU run (next line runs keras on GPU)
    # plaidml.keras.install_backend()

def sr_up(image):
    t0 = time.time()
    result = sr.upsample(image)
    t1 = time.time()
    dt = t1 - t0
    return result, dt

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


def sharpen_color(image, sigma, strength):
    r1 = np.zeros_like(image)
    dt2 = 0
    for i in range(3):
        # r1[:, :, i], dt = sharpen(result[:, :, i], 1, 0.2)  # max strength if before upscale
        r1[:, :, i], dt = sharpen_channel(image[:, :, i], sigma, strength)
        dt2 = dt2 + dt
    
    result = r1
    return result, dt2


def process_pipeline(image, i_name, dt_set):
    if dt_set.start:
        o_ext = V_O_EXT
    else:
        o_ext = I_O_EXT
    
    upscale_lat = ''
    presharp_lat = ''
    postsharp_lat = ''

    pipe_t0 = time.time()
    result = image
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Unsharp-masking (very light)
    if not dt_set.no_presharp:
        result, dt = sharpen_color(result, 1, 0.2)
        presharp_lat = 'presharp {:.{prec}f} sec'.format(dt, prec=2)
        
    # Upscale the image
    result, dt = sr_up(result)
    upscale_lat = 'upscale {:.{prec}f} sec'.format(dt, prec=2)

    # Post unsharp custom stength
    if dt_set.postsharpen > 0:
        result, dt = sharpen_color(result, 1, dt_set.postsharpen)
        postsharp_lat = 'postsharp {:.{prec}f} sec'.format(dt, prec=2)

    # Save final frame    
    workdir = os.getcwd()
    o_path = '{}/{}/{}.{}'.format(workdir, M_LABEL, i_name, imw_code[o_ext][1])
    was_wrtn = cv2.imwrite(o_path, result, imw_code[o_ext][0])

    if was_wrtn:
        out_hr = '{}_x{}: {}'.format(PRE_BUILT[MOD_NAME], UP_SIZE, i_name)
        tot_lat = time.time() - pipe_t0
        print('{}\t{}\t{}\t{}\ttot_lat: {:.{prec}f}'.
        format(out_hr, upscale_lat, presharp_lat, postsharp_lat, tot_lat, prec=2))


def ext_based_workflows(dt_set):
    file_pattern = dt_set.file

    if dt_set.start:
        v_start = int(dt_set.start[0])
    else:
        v_start = 0
    
    if dt_set.end:
        v_stop = int(dt_set.end[0])
    else:
        v_stop = 0
    
    for fp in file_pattern:
        ext = fp.split('.')[-1].lower()

        if ext in ['jpg', 'jpeg', 'png']:
            # workflow for individual image_files
            image = cv2.imread(fp)
            i_name = '.'.join(fp.split('.')[:-1])
            process_pipeline(image, i_name, dt_set)

        elif ext in ['mp4', 'm4v', 'mkv', 'avi']:
            # workflow for pulling images out of video files (adj requesting frame 1 = 0 secs)
            try:
                v_stream = cv2.VideoCapture(fp)
                tot_frames = v_stream.get(cv2.CAP_PROP_FRAME_COUNT)
                if v_stop == 0:
                    v_stop = tot_frames
                # fps = v_stream.get(cv2.CAP_PROP_FPS)
                # tot_time_sec = tot_frames/fps
                # v_start_sec = (v_start-1)/fps
                # start_time_code = v_start_sec/tot_time_sec
                # zero_idx_time_code = (v_start-1)/tot_frames
                v_stream.set(cv2.CAP_PROP_POS_FRAMES, v_start-1)
                grab_frame_stat, img = v_stream.read()
                if not grab_frame_stat:
                    print('Unable to decode video frame from [{}] at frame {}'.format(fp, v_start))
                    if v_stream.isOpened():
                        print('Did you forget to install ffmpeg and its dependencies before building opencv?')
                    else:
                        print('v_stream not even able to open')
                count = v_start
                while grab_frame_stat:
                    if count > v_stop:
                        break
                    elif count >= v_start:
                        i_name = '{:0>{width}}'.format(count, width=7)
                        process_pipeline(img, i_name, dt_set)
                    else:
                        # sequential processing is slow, this section deprecated with set function above
                        print('\r{:>0{width}}'.format(count, width=7))
                    grab_frame_stat, img = v_stream.read()
                    count += 1
                v_stream.release()
            except ZeroDivisionError:
                print('Could not get video duration.')
                pass


def get_cli_args(args):
    parser = argparse.ArgumentParser(description='Upconvert a video or image(s) using NN SR.')
    parser.add_argument('-s', '--start', nargs=1, required=False,
        help='integer starting frame (videos start at 1 not 0)')
    parser.add_argument('-e', '--end', nargs=1, required=False,
        help='integer ending frame (videos start at 1 not 0)')
    parser.add_argument('-nx0', '--no-presharp', default=False)
    parser.add_argument('-ps', '--postsharpen', nargs=1, default='0', help='unsharp strength 0=off 0.7 to 2 make sense')
    parser.add_argument('file', nargs='*',
        help='file_name or file_name_pattern')
    ui_cfg = parser.parse_args(args)
    try:
        ui_cfg.postsharpen = float(ui_cfg.postsharpen[0])
    except:
        raise
    return ui_cfg


def main(args):
    dt_set = get_cli_args(args)

    set_sr_model(MOD_NAME, UP_SIZE)

    # Create a subfolder for output using model label name with x factor
    if not os.path.isdir(M_LABEL):
        os.mkdir(M_LABEL)

    ext_based_workflows(dt_set)
    
    return M_LABEL


if __name__ == "__main__":
    main(sys.argv[1:])
