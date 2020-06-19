#!/usr/bin/env python3

import numpy as np
import time
import cv2
from cv2 import dnn_superres
import plaidml.keras
import imutils
import glob
import os

cv2.ocl.setUseOpenCL(True)


def save_jpg_std(result):
    cv2.imwrite('{}/{}/{}.jpg'.format(workdir, m_label, i_name), result,
                [int(cv2.IMWRITE_JPEG_QUALITY), 87])


def save_j90_std(result):
    cv2.imwrite('{}/{}/{}.jpg'.format(workdir, m_label, i_name), result,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_png_std(result):
    cv2.imwrite('{}/{}/{}.png'.format(workdir, m_label, i_name), result)


def upscale_frame(image, i_name, o_ext):
    timings = np.array([])

    # Upscale the image
    t0 = time.time()
    result = sr.upsample(image)
    t1 = time.time()
    dt = t1 - t0
    timings = np.append(timings, dt)

    if o_ext == 'jpg':
        save_jpg_std(result)
    elif o_ext == 'j90':
        save_j90_std(result)
    elif o_ext == 'png':
        save_png_std(result)

    print('{}_x{}: {} took {:.{prec}f} sec | avg {:.{prec}f} sec'.
          format(prebuilt[modname], upsize,
                 i_name,
                 dt,
                 np.average(timings),
                 prec=2))

# cv2.imshow("Image", image)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    modname = 'edsr'
    upsize = 4
    file_pattern = 'a9d46429486d82122cb69c64148efc63.jpg'
    homedir = os.getenv('HOME')
    workdir = '{}/Downloads/'.format(homedir)
    v_start = 757
    v_stop = 5000

    prebuilt = {
        'edsr': 'EDSR',
        'espcn': 'ESPCN',
        'lapsrn': 'LapSRN',
        'fsrcnn': 'FSRCNN',
        'fsrcnn-sm': 'FSRCNN-small'
    }

    m_path = "{}_x{}.pb".format(prebuilt[modname], upsize)
    m_label = m_path.split('.')[0]

    sr.readModel(m_path)
    sr.setModel(modname, upsize)

    # CUDA driver and toolkit support problems on macOS mojave
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # TODO: FSRCNN has been ported to tf, will need to port tf to keras for GPU run (next line runs keras on GPU)
    # plaidml.keras.install_backend()

    os.chdir(workdir)

    if not os.path.isdir(m_label):
        os.mkdir(m_label)

    ext = file_pattern.split('.')[-1].lower()

    if ext in ['jpg', 'jpeg', 'png']:
        img_files = glob.glob(file_pattern)
        img_files.sort()

        for img in img_files:
            image = cv2.imread(img)
            i_name = '.'.join(img.split('.')[:-1])
            upscale_frame(image, i_name, 'jpg')

    elif ext in ['mp4', 'm4v']:
        try:
            v_stream = cv2.VideoCapture(file_pattern)
            frame_read, img = v_stream.read()
            count = 1
            while frame_read:
                if count >= v_start and count <= v_stop:
                    i_name = '{:0>{width}}'.format(count, width=7)
                    upscale_frame(img, i_name, 'j90')
                frame_read, img = v_stream.read()
                count += 1
        except:
            pass


