#!/usr/bin/env python3

import cv2
from cv2 import dnn_superres
from scipy.ndimage.filters import median_filter
from tqdm import tqdm
import numpy as np
import plaidml.keras
import time
import os
import sys
import glob
import argparse
import logging
import signal

my_pid = os.getpid()

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
# ffv1 writes a lossless mkv video output file while avc1 is lossy but faster & smaller
imw_code = {
    'jpg': ([cv2.IMWRITE_JPEG_QUALITY, 87], 'jpg'),
    'j90': ([cv2.IMWRITE_JPEG_QUALITY, 90], 'jpg'),
    'j100': ([cv2.IMWRITE_JPEG_QUALITY, 100], 'jpg'),
    'png': ([cv2.IMWRITE_PNG_COMPRESSION, 1], 'png'),
    'png3': ([cv2.IMWRITE_PNG_COMPRESSION, 4], 'png'),
    'png9': ([cv2.IMWRITE_PNG_COMPRESSION, 9], 'png'),
    'wpll': ([cv2.IMWRITE_WEBP_QUALITY, 101], 'webp'),
    'mp4v': (cv2.VideoWriter_fourcc(*'mp4v'), 'mp4'),
    'avc1': (cv2.VideoWriter_fourcc(*'avc1'), 'mp4'),
    'avc3': (cv2.VideoWriter_fourcc(*'avc3'), 'mkv'),
    'hev1': (cv2.VideoWriter_fourcc(*'hev1'), 'mkv'),
    'ffv1': (cv2.VideoWriter_fourcc(*'FFV1'), 'mkv'),
    'hfyu': (cv2.VideoWriter_fourcc(*'HFYU'), 'mkv')
}
V_CODEC = ['mp4v', 'avc1', 'avc3', 'hev1', 'ffv1', 'hfyu']
V_O_EXT = 'png9'
I_O_EXT = 'jpg'
I_O_DIR = M_LABEL

# Create a global SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Load ONNX network
srpath = '{}/{}'.format(M_FQP, 'super-resolution-10.onnx')
net = cv2.dnn.readNetFromONNX(srpath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) 
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Handy debugging
# before = img.shape()
# after = result.shape()

vout = None

def terminateProcess(signalNumber, frame):
    try:
        vout.release()
        msg = 'Signal {} recieved and gracefully saved video output'.format(signalNumber)
    except:
        msg = 'Signal {} recieved and no video output to close'.format(signalNumber)
    
    logging.info(msg)
    print(msg)
    sys.exit
    os.kill(my_pid, signal.SIGQUIT)

def plaidml_decorator(func):
    def wrapper(*args, **kwargs):
        """ FSRCNN has been ported to tf, will need to port tf to keras 
        for GPU run (next line runs keras on GPU)
        TODO: Not tested yet!!
        """
        plaidml.keras.install_backend()
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def cuda_decorator(func):
    def wrapper(*args, **kwargs):
        """ TODO: currently unable to compile opencv for CUDA on macOS 10.15.x
        CUDA driver and toolkit support problems on macOS mojave
        TODO: Not tested yet!!
        """
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def time_it_decorator(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        dt = time.time() - t0
        return func(*args, **kwargs), dt
    return wrapper

def set_sr_model(model, scale):
    """ Load DNN SuperResolution model
    Wrap this function with the @cuda_decorator or @plaidml_decorator to
    run SR on GPU.. or selectively run with call wrapping
    """
    # m_fname = '{}_x{}.pb'.format(PRE_BUILT[model], scale)
    # m_fqp = os.path.dirname(os.path.realpath(__file__))
    # m_fqn = '{}/{}'.format(m_fqp, m_fname)
    sr.readModel(M_FQN)
    sr.setModel(model, scale)    

def sharpen_channel_laplacian(image, sigma, strength):
    """ Sharpen a 1-D image array (grayscale or single color)
    """
    mf_image = median_filter(image, sigma)
    lap = cv2.Laplacian(mf_image, cv2.CV_64F)
    result = image - strength * lap
    """ Even mild sharpening always causes clipping
    clip_hi = np.amax(result)
    clip_lo = np.amin(result)
    if clip_lo < 0 or clip_hi > 255:
        logging.warning('sharpen filter cliping: hi_val {} lo_val {}'.format(
            clip_hi, clip_lo ))
    """
    result[result > 255] = 255
    result[result < 0] = 0
    return result

def sharpen_color_laplacian(image, sigma, strength):
    """ sharpen a 3 color image array one channel at a time
    """
    result = np.zeros_like(image)
    for i in range(3):
        result[:, :, i] = sharpen_channel_laplacian(image[:, :, i], sigma, strength)    
    return result

def sharpen_color_guasian(image, alpha, beta, gamma):
    # alpha = 1.5
    # beta = -0.5
    # gamma = 0
    # FIXME: Not implemented correctly... just blurring images
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    result = cv2.addWeighted(blur, alpha, image, beta, gamma)
    return result

def sharpen_hsv_saturation(image, sigma, strength):
    """ Described in medical imaging journals to help medical diags.
    OpenCV hue ranges are [0-179] while saturation and values are [0-255]
    cv2.COLOR_BGR2HSV_FULL scales hue to [0-255] for compatibility outside
    of cv2 tool.
    # FIXME: Not very applicable to movies and home photo sharpening
    """ 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = sharpen_channel_laplacian(hsv_image[:, :, 1], sigma, strength)
    result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return result

def sharpen_kern_filter(image):
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    result = cv2.filter2D(image,-1,filter)
    return result

def blur_detection(image):
    """ Variance of the Laplacian gives a good score to determine how
    sharp an image might be. Observations indicate 000s is HD while
    most scores below 80-100 are often low quality.
    Future work: Consider a normalized score rel img size?
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

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
    
    lpc_score = blur_detection(result)
    stage_dts.append('origLpcVar:{:.0f}'.format(round(lpc_score)))

    """ crop black boarders """
    if dt_set.crop:
        l, r, t, b = dt_set.crop
        h, w, chan = result.shape
        start_x = int(l)
        stop_x = w - int(r)
        start_y = int(t)
        stop_y = h - int(b)
        logging.debug(
            'crop frame: y0:{} y1:{} x0:{} x1:{}'.format(
                start_y, stop_y, start_x, stop_x)
            )
        result = result[start_y:stop_y, start_x:stop_x]

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
        run_decorator = time_it_decorator(cv2.fastNlMeansDenoisingColored)
        result, dt = run_decorator(result, None, hlc, hlc, 7, 21)
        stage_dts.append('nlmeans {:.2f}s'.format(dt))

    """ try inpainting edges as a way to correct """
    # edge = cv2.Canny(image, 50, 100)
    # mask = edge
    # result = cv2.inpaint(image, mask, 3, flags=cv2.INPAINT_NS)
    # # cv2.imshow('mask', mask)

    if dt_set.presharp > 0:
        # pre SR shouldn't use more the 0.1
        run_decorator = time_it_decorator(sharpen_color_laplacian)
        result, dt = run_decorator(result, 1, dt_set.presharp)
        stage_dts.append('sharpen {:.{prec}f}s'.format(dt, prec=2))
    
    """ ese = expand shrink expand... experiment with lanczos4 sharpening """
    if dt_set.ese:
        orig_y, orig_x, chans = result.shape
        run_decorator = time_it_decorator(sr.upsample)
        result, dt = run_decorator(result)
        run_decorator = time_it_decorator(cv2.resize)
        result, dt2 = run_decorator(result, (orig_x, orig_y), interpolation=cv2.INTER_LANCZOS4)
        stage_dts.append('ESE {:.2f}s'.format(dt+dt2))
        lpc_score = blur_detection(result)
        stage_dts.append('lpcVar:{:.0f}'.format(lpc_score))    

    """ Upscale the image """
    if dt_set.sr10:
        blob = cv2.dnn.blobFromImage(result, swapRB=True, crop=False)
        net.setInput(blob)
        result = net.forward()
    else:
        # result, dt = sr_up(result)
        run_decorator = time_it_decorator(sr.upsample)
        result, dt = run_decorator(result)
        stage_dts.append('SupRes {:.2f}s'.format(dt))
        lpc_score = blur_detection(result)
        stage_dts.append('lpcVar:{:.0f}'.format(lpc_score))

    """ Post nlmeans denoising """
    if dt_set.postdenoise > 0:
        run_decorator = time_it_decorator(cv2.fastNlMeansDenoisingColored)
        h = dt_set.postdenoise
        result, dt = run_decorator(result, None, h, h, 7, 21)
        stage_dts.append('nlmeans {:.{prec}f}s'.format(dt, prec=2))

    # # try inpainting edges as a way to denoise?
    # edge = cv2.Canny(result, 50, 100)
    # mask = edge
    # result = cv2.inpaint(result, mask, 3, flags=cv2.INPAINT_NS)
    # cv2.imshow('mask', mask)

    # Post unsharp custom stength
    if dt_set.postsharpen > 0:
        run_decorator = time_it_decorator(sharpen_color_laplacian)
        result, dt = run_decorator(result, 1, dt_set.postsharpen)
        stage_dts.append('sharpen {:.{prec}f}s'.format(dt, prec=2))
        lpc_score = blur_detection(result)
        stage_dts.append('lpcVar:{:.0f}'.format(round(lpc_score)))
    if dt_set.postsharpen2 > 0:
        run_decorator = time_it_decorator(sharpen_kern_filter)
        # result, dt = run_decorator(result, 1, dt_set.postsharpen2)
        # stage_dts.append('sharpen_Sat {:.{prec}f}s'.format(dt, prec=2))
        result, dt = run_decorator(result)
        stage_dts.append('sharpen_kernel {:.{prec}f}s'.format(dt, prec=2))
        lpc_score = blur_detection(result)
        stage_dts.append('lpcVar:{:.0f}'.format(round(lpc_score)))

    # cv2.imshow('post-sharpen', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save final frame    
    try:
        # try using video output object and fall back to imwrite
        dsize = dt_set.dsize
        # was_wrtn = dt_set.vout.write(result)
        was_wrtn = vout.write(cv2.resize(result, dsize, interpolation=cv2.INTER_CUBIC))
        # TODO: could use laplacian variance to test bicubic vs lanczos4 ?
        if was_wrtn is None:
            was_wrtn = True
    except AttributeError:
        workdir = os.getcwd()
        o_path = '{}/{}/{}.{}'.format(workdir, dt_set.out_dir, i_name, imw_code[o_ext][1])
        was_wrtn = cv2.imwrite(o_path, result, imw_code[o_ext][0])

    if was_wrtn and dt_set.log in ['info', 'INFO']:
        out_hr = '[{:.{prec}f}%] {}_x{}: {}'.format(
            dt_set.pct_done, PRE_BUILT[MOD_NAME], UP_SIZE, i_name, prec=1)
        tot_lat = time.time() - pipe_t0
        logging.info('{} {} tot_lat: {:.{prec}f}s'.
        format(out_hr, stage_dts, tot_lat, prec=2))


def ext_based_workflows(dt_set):
    file_pattern = dt_set.file

    if dt_set.start:
        v_start = int(dt_set.start)
    else:
        v_start = 1
    
    if dt_set.end:
        v_stop = int(dt_set.end)
    else:
        v_stop = 0
    
    p_frame_thresh = 300000
    back_scan_offset = 2

    print('Running as PID:', my_pid)
    logging.info(' **** RUN STARTED with PID: {} ****'.format(my_pid))
    
    pbar0 = tqdm(total = len(file_pattern), unit=' files', mininterval=1.0)

    for fp in file_pattern:
        ext = fp.split('.')[-1].lower()
        fnpfx = fp.split('.')[:-1][0]

        if ext in ['jpg', 'jpeg', 'png']:
            # workflow for individual image_files
            if os.path.isfile(fp):
                image = cv2.imread(fp)
                i_name = '.'.join(fp.split('.')[:-1])
                process_pipeline(image, i_name, dt_set)
            else:
                logging.warning('file: {} does not exist'.format(fp))
                print('file: {} does not exist'.format(fp))
                sys.exit()

        elif ext in ['mp4', 'm4v', 'mkv', 'avi']:
            # workflow for pulling images out of video files (adj requesting frame 1 = 0 secs)
            try:
                v_stream = cv2.VideoCapture(fp)
                tot_frames = int(v_stream.get(cv2.CAP_PROP_FRAME_COUNT))
                if v_stop == 0:
                    v_stop = tot_frames
                
                # Setup video writer object to receive new SR frames
                o_ext = dt_set.out_ext
                if o_ext in V_CODEC:
                    fps = v_stream.get(cv2.CAP_PROP_FPS)
                    # TODO: Need PAR input to do SAR * PAR = DAR (display AR) for resizing

                    # tot_time_sec = tot_frames/fps
                    # v_start_sec = (v_start-1)/fps
                    # start_time_code = v_start_sec/tot_time_sec
                    # zero_idx_time_code = (v_start-1)/tot_frames
                    
                    fourcc = imw_code[o_ext][0]
                    i_name = '{}_{}_{:0>{width}}-{:0>{width}}.{}'.format(
                        fnpfx, 
                        o_ext.upper(), 
                        v_start, v_stop, 
                        imw_code[o_ext][1],
                        width=len(str(v_stop))
                        )
                    o_path = '{}/{}/{}'.format(os.getcwd(), dt_set.out_dir, i_name)
                    lx, rx, ty, by = dt_set.crop
                    ax, ay = dt_set.par
                    sr_x = int((v_stream.get(cv2.CAP_PROP_FRAME_WIDTH) - int(lx) - int(rx)) * UP_SIZE)
                    sr_y = int((v_stream.get(cv2.CAP_PROP_FRAME_HEIGHT) - int(ty) - int(by)) * UP_SIZE)
                    
                    # TODO: need to create checks as this assumes the sr_y > 1080
                    # new_y = 1080
                    # new_x = int(2 * round(((sr_x * new_y)/sr_y)/2))
                    
                    new_x = 1920
                    # new_y = int(round(((new_x * sr_y)/sr_x)/2) * 2)
                    new_y = (((new_x * sr_y)/sr_x)*int(ay))/int(ax)
                    new_y = int(round(new_y/2)*2)

                    dsize = (new_x, new_y)
                    
                    global vout
                    
                    vout = cv2.VideoWriter(o_path, fourcc, fps, dsize, True)
                    vout.set(cv2.VIDEOWRITER_PROP_NSTRIPES, -1)
                    vout.set(cv2.VIDEOWRITER_PROP_QUALITY, 100.)
                    logging.info(' openned: {}'.format(o_path))
                    # dt_set.vout = vout
                    dt_set.dsize = dsize

                v_stream.set(cv2.CAP_PROP_POS_FRAMES, v_start-1)
                grab_frame_stat, img = v_stream.read()
                if not grab_frame_stat:
                    logging.warning('Unable to decode video frame from [{}] at frame {}'.format(fp, v_start))
                    if v_stream.isOpened():
                        logging.warning('Did you install ffmpeg before building opencv?')
                    else:
                        logging.warning('v_stream not even able to open')
                if v_start > 1:
                    
                    """ Test prev frame to see if current frame is P frame? """
                    v_stream.set(cv2.CAP_PROP_POS_FRAMES, v_start - back_scan_offset)
                    grab_frame_stat, img_pre = v_stream.read()
                    while grab_frame_stat:
                        diff = cv2.absdiff(img, img_pre)
                        non_zero_count = np.count_nonzero(diff)

                        # TODO: scan and detect which frames have scene transitions..
                        # # TODO: study the following during hist shifts for hdiff threshold
                        # nbins = 64
                        # min = np.minimum(img, img_pre)
                        # max = np.maximum(img, img_pre)
                        # hist1, _ = np.histogram(img, range(min, max), bins=nbins)
                        # hist2, _ = np.histogram(img_pre, range(min, max), bins=nbins)
                        # hdiff = np.absolute(hist1 - hist2)
                        
                        # TODO: select diff SR model for dark upscale + enhanced contrast
                        # TODO: re-analyze blur, pre-post filter settings on scene changes?
                        # TODO: create a transfer-learning repo w lables and features?
                        # Scene change can be detected with histogram median shifts
                        # And light/dark scenes can be detected with shift in normalized
                        # median luma level shifting
                                             
                        if v_start - back_scan_offset < 0:
                            break
                        elif non_zero_count > p_frame_thresh:
                            """ This method of guessing P-frames is a borrowed hack
                            That has shown to work and saved a bit of frustration
                            """
                            back_scan_offset += 1
                            img = img_pre
                            v_stream.set(cv2.CAP_PROP_POS_FRAMES, v_start - back_scan_offset)
                            grab_frame_stat, img_pre = v_stream.read()
                        else:
                            break
                    
                    """ math-wise the if != 2 does the same thing as == 2 block """
                    if back_scan_offset == 2:
                        """ if non-P frame reset to original start """
                        v_stream.set(cv2.CAP_PROP_POS_FRAMES, v_start-1)
                        grab_frame_stat, img = v_stream.read()
                    else:
                        """ skip back and scan forward """
                        m1 = 'User selected frame might be a P-Frame:'
                        m2 = 'pre-scanned {} frames'.format(back_scan_offset)
                        logging.info(m1 + m2)
                        new_start = v_start - (back_scan_offset - 1)
                        v_stream.set(cv2.CAP_PROP_POS_FRAMES, new_start)
                        for trash in range(new_start, v_start, 1):
                            grab_frame_stat, img = v_stream.read()
                """ start normal frame retrieval process """
                count = v_start
                tot_xf = v_stop - v_start
                pbar1 = tqdm(total = tot_xf + 1, unit=' frames', mininterval=1.0)
                while grab_frame_stat:
                    if count > v_stop:
                        break
                    elif count >= v_start:
                        i_name = '{:0>{width}}'.format(count, width=len(str(v_stop)))
                        dt_set.pct_done = float(count - v_start)*100/tot_xf
                        process_pipeline(img, i_name, dt_set)
                    else:
                        # sequential processing is slow, this section deprecated with set function above
                        logging.debug('unknown frame loop {:>0{width}}'.format(count, width=7))
                    grab_frame_stat, img = v_stream.read()
                    count += 1
                    pbar1.update(1)
                pbar1.close()
                v_stream.release()
                try:
                    vout.release()
                except (UnboundLocalError, AttributeError):
                    pass               
            except ZeroDivisionError:
                logging.warning('Could not get video duration.')
                pass
        pbar0.update(1)
    pbar0.close()


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
    parser.add_argument('-cr', '--crop', nargs=4, required=False, 
        default=['0','0','0','0'],
        help='Lft Rgt Top Bot pixels to cut')
    parser.add_argument('-par', '--par', nargs=2, required=False,
        default=['1', '1'],
        help='x y')
    parser.add_argument('-x0', '--presharp', nargs=1, default='0',
        help='pre upscale sharpening strength 0=off 0.1 to 0.2 make sense')
    parser.add_argument('-nl0', '--pre-denoise', nargs=1, default=0,
        help='pre upscale nlmeans denoising hC=hL integer')
    parser.add_argument('-ese', action='store_true')
    parser.add_argument('-sr10', action='store_true')
    parser.add_argument('-nl1', '--postdenoise', nargs=1, default=0,
        help='post upscale nlmeans denoising hC=hL integer')        
    parser.add_argument('-x1', '--postsharpen', nargs=1, default='0', 
        help='unsharp RGB strength 0=off 0.7 to 2 make sense')
    parser.add_argument('-x2', '--postsharpen2', nargs=1, default='0', 
        help='unsharp HSV saturation strength 0=off 0.7 to 2 make sense')
    parser.add_argument('-log', '--log', nargs=1, default='WARNING', 
        choices=['debug', 'info', 'warning'])
    parser.add_argument('file', nargs='*',
        help='file_name or file_name_pattern')
    try:
        ns = parser.parse_args(args)

        # Convenience: unpack ns scalar entries from lists to scalars except ['files']
        for k, v in ns.__dict__.items():
            if isinstance(v, list) and k != 'file':
                if len(v) == 1:
                    ns.__dict__.update({k: v[0]})

        # For args with defaults add kv ease passing args in pipeline
        ns.kw_cus_arg = {'o_ext': ns.out_ext}
        ns.kw_cus_arg.update({'o_dir': ns.out_dir})

    # try:
        ns.postsharpen = float(ns.postsharpen)
        ns.postsharpen2 = float(ns.postsharpen2)
        ns.presharp = float(ns.presharp)
        ns.pre_denoise = int(ns.pre_denoise)
        ns.postdenoise = int(ns.postdenoise)
    except:
        raise

    return ns

def main(args):
    if len(args) < 1:
        args.append('-h')
    # get user cli options with argparse module or exit
    ns = get_cli_args(args)
    logging.basicConfig(
        filename='dnn_superres.log', 
        format='%(asctime)s %(levelname)s:%(message)s', 
        level=ns.log.upper()
        )
    # @decorate this target function or manually wrap next line for gpu
    set_sr_model(MOD_NAME, UP_SIZE)
    
    # Create a subfolder for output
    o_dir = ns.kw_cus_arg['o_dir']
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)

    # Start the image file or video file workflow
    ext_based_workflows(ns)
    
    # Let the calling function know where outputs were written
    return o_dir

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminateProcess)
    signal.signal(signal.SIGINT, terminateProcess)
    
    main(sys.argv[1:])