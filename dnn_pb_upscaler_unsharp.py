#!/usr/bin/env python3

import cv2
from cv2 import dnn_superres
from scipy.ndimage.filters import median_filter
import numpy as np
import plaidml.keras
import time
import os
import sys
import glob
import argparse

cv2.ocl.setUseOpenCL(True)

MOD_NAME = 'lapsrn'
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
I_O_DIR = M_LABEL

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

def time_it(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        dt = time.time() - t0
        return func(*args, **kwargs), dt
    return wrapper

def sharpen_channel_laplacian(image, sigma, strength):
    """ sharpen a monochromatic image array
    """
    mf_image = median_filter(image, sigma)
    lap = cv2.Laplacian(mf_image, cv2.CV_64F)
    result = image - strength * lap
    # clip_hi = np.amax(result)
    # clip_lo = np.amin(result)
    # if clip_lo < 0 or clip_hi > 255:
    #     print('sharpen filter cliping: hi_val {} lo_val {}'.format(
    #         clip_hi, clip_lo ))
    result[result > 255] = 255
    result[result < 0] = 0
    return result

def sharpen_color_laplacian(image, sigma, strength):
    """ sharpen a 3 color image array
    """
    result = np.zeros_like(image)
    for i in range(3):
        result[:, :, i] = sharpen_channel_laplacian(image[:, :, i], sigma, strength)    
    return result

def sharpen_color_guasian(image, alpha, beta, gamma):
    # alpha = 1.5
    # beta = -0.5
    # gamma = 0
    blur = cv2.GaussianBlur(image, cv2.Size(0, 0), 3)
    result = cv2.addWeighted(blur, alpha, image, beta, gamma)
    return result

def process_pipeline(image, i_name, dt_set):
    # if dt_set.start:
    #     o_ext = V_O_EXT
    # else:
    #     o_ext = I_O_EXT
    o_ext = dt_set.out_ext
    
    upscale_lat = ''
    presharp_lat = ''
    postsharp_lat = ''
    stage_dts = []
    pipe_t0 = time.time()
    result = image
    
    """ autocrop black boarders """
    if dt_set.autocrop:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # x, y, w, h = cv2.boundingRect(contours[0]) # new boundary
        
        c = max(contours, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        # print('x:{} y:{} w:{} h:{}'.format(extLeft[0], extTop[1], extRight[0], extBot[1]))
        x, y, w, h = [extLeft[0], extTop[1], extRight[0], extBot[1]]
        print('autocrop frame: x:{} y:{} w:{} h:{}'.format(x, y, w, h))
        result = result[y:y+h, x:x+w]    # crop

    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('thresh', thresh)
    
    # Inpaint broken edges? didn't really help
    # result = cv2.inpaint(result, thresh, 3, flags=cv2.INPAINT_TELEA)
    # cv2.imshow('fix-edges', result)
    
    # gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)
    
    if dt_set.pre_denoise > 0:
        # pre SR denoising 1, 1 should just deblock or rm small noise
        hlc = int(dt_set.pre_denoise)
        run_decorator = time_it(cv2.fastNlMeansDenoisingColored)
        result, dt = run_decorator(result, None, hlc, hlc, 7, 21)
        stage_dts.append('nlmeans {:.2f}s'.format(dt))

    """ try inpainting edges as a way to correct """
    # edge = cv2.Canny(image, 50, 100)
    # mask = edge
    # result = cv2.inpaint(image, mask, 3, flags=cv2.INPAINT_NS)
    # # cv2.imshow('mask', mask)

    if dt_set.presharp > 0:
        # pre SR shouldn't use more the 0.1
        run_decorator = time_it(sharpen_color_laplacian)
        result, dt = run_decorator(result, 1, dt_set.presharp)
        stage_dts.append('sharpen {:.{prec}f}s'.format(dt, prec=2))
    
    """ Upscale the image """
    # result, dt = sr_up(result)
    run_decorator = time_it(sr.upsample)
    result, dt = run_decorator(result)
    stage_dts.append('upscale {:.2f}s'.format(dt))

    # try minimal post SR denoising as well
    # result = cv2.fastNlMeansDenoisingColored(result, None, 1, 1, 7, 21)

    # # try inpainting edges as a way to denoise?
    # edge = cv2.Canny(result, 50, 100)
    # mask = edge
    # result = cv2.inpaint(result, mask, 3, flags=cv2.INPAINT_NS)
    # cv2.imshow('mask', mask)

    # Post unsharp custom stength
    if dt_set.postsharpen > 0:
        run_decorator = time_it(sharpen_color_laplacian)
        result, dt = run_decorator(result, 1, dt_set.postsharpen)
        stage_dts.append('sharpen {:.{prec}f}s'.format(dt, prec=2))

    # cv2.imshow('post-sharpen', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save final frame    
    workdir = os.getcwd()
    o_path = '{}/{}/{}.{}'.format(workdir, dt_set.out_dir, i_name, imw_code[o_ext][1])
    was_wrtn = cv2.imwrite(o_path, result, imw_code[o_ext][0])

    if was_wrtn:
        out_hr = '[{:.{prec}f}%] {}_x{}: {}'.format(
            dt_set.pct_done, PRE_BUILT[MOD_NAME], UP_SIZE, i_name, prec=1)
        tot_lat = time.time() - pipe_t0
        print('{} {} tot_lat: {:.{prec}f}s'.
        format(out_hr, stage_dts, tot_lat, prec=2))


def ext_based_workflows(dt_set):
    file_pattern = dt_set.file

    if dt_set.start:
        v_start = int(dt_set.start)
    else:
        v_start = 0
    
    if dt_set.end:
        v_stop = int(dt_set.end)
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
                        dt_set.pct_done = float(count - v_start)/(v_stop - v_start)*100
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
    parser.add_argument('-d', '--out-dir', nargs=1, required=False,
        default=I_O_DIR,
        help='string to override default output subfolder name')
    parser.add_argument('-ext', '--out-ext', nargs=1, required=False, 
        default=V_O_EXT,
        choices=imw_code.keys(),
        help='string to define output type')
    parser.add_argument('-ac', '--autocrop', action='store_true')
    parser.add_argument('-x0', '--presharp', nargs=1, default='0',
        help='pre upscale sharpening strength 0=off 0.1 to 0.2 make sense')
    parser.add_argument('-nl0', '--pre-denoise', nargs=1, default=0,
        help='pre upscale nlmeans denoising hC=hL integer')
    parser.add_argument('-x1', '--postsharpen', nargs=1, default='0', 
        help='unsharp strength 0=off 0.7 to 2 make sense')
    parser.add_argument('file', nargs='*',
        help='file_name or file_name_pattern')
    ns = parser.parse_args(args)

    for k, v in ns.__dict__.items():
        if isinstance(v, list) and k != 'file':
            if len(v) == 1:
                ns.__dict__.update({k: v[0]})

    # For args with defaults to friendly kv
    ns.kw_cus_arg = {'o_ext': ns.out_ext}
    ns.kw_cus_arg.update({'o_dir': ns.out_dir})

    try:
        ns.postsharpen = float(ns.postsharpen)
        ns.presharp = float(ns.presharp)
        ns.pre_denoise = int(ns.pre_denoise)
    except:
        raise
    return ns


def main(args):
    ns = get_cli_args(args)

    set_sr_model(MOD_NAME, UP_SIZE)
    
    o_dir = ns.kw_cus_arg['o_dir']
    # Create a subfolder for output using model label name with x factor
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)

    ext_based_workflows(ns)
    
    return o_dir

if __name__ == "__main__":
    main(sys.argv[1:])
