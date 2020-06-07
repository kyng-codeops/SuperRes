import numpy as np
import time
import cv2
from cv2 import dnn_superres
import plaidml.keras
import imutils
import glob
import os
from scipy.ndimage.filters import median_filter

cv2.ocl.setUseOpenCL(True)


def save_jpg_std(result, i_name):
    cv2.imwrite('{}/{}/{}.jpg'.format(workdir, m_label, i_name), result,
                [int(cv2.IMWRITE_JPEG_QUALITY), 87])


def save_j90_std(result, i_name):
    cv2.imwrite('{}/{}/{}.jpg'.format(workdir, m_label, i_name), result,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_png_std(result, i_name):
    cv2.imwrite('{}/{}/{}.png'.format(workdir, m_label, i_name), result)


def sr_up(image):
    t0 = time.time()
    result = sr.upsample(image)
    t1 = time.time()
    dt = t1 - t0
    return result, dt


def sharpen(image, sigma, strength):
    t0 = time.time()

    mf_image = median_filter(image, sigma)
    lap = cv2.Laplacian(mf_image, cv2.CV_64F)
    result = image - strength * lap
    result[result > 255] = 255
    result[result < 0] = 0

    t1 = time.time()
    dt = t1 - t0
    return result, dt


def process_pipeline(image, i_name, o_ext):
    timings_sr = np.array([])
    timings_us = np.array([])

    result = image

    # Unsharp-masking
    r1 = np.zeros_like(result)
    dt2 = 0
    for i in range(3):
        r1[:, :, i], dt = sharpen(result[:, :, i], 1, 0.2)
        dt2 = dt2 + dt
    # result, dt2 = sharpen(result, 2, 1.)
    result = r1
    timings_us = np.append(timings_us, dt2)

    # Upscale the image
    result, dt1 = sr_up(result)
    timings_sr = np.append(timings_sr, dt1)

    if o_ext == 'jpg':
        save_jpg_std(result, i_name)
    elif o_ext == 'j90':
        save_j90_std(result, i_name)
    elif o_ext == 'png':
        save_png_std(result, i_name)

    out_hr = '{}_x{}: {}'.format(prebuilt[modname], upsize, i_name)
    upscale_out = 'upscale {:.{prec}f} sec | avg {:.{prec}f} sec'.format(dt1, np.average(timings_sr), prec=2)
    unsharp_out = 'unsharp {:.{prec}f} sec | avg {:.{prec}f} sec'.format(dt2, np.average(timings_us), prec=2)
    print('{}\n\t\t{}\t\t{}'.format(out_hr, upscale_out, unsharp_out))

# cv2.imshow("Image", image)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def ext_based_workflows():
    ext = file_pattern.split('.')[-1].lower()

    if ext in ['jpg', 'jpeg', 'png']:
        img_files = glob.glob(file_pattern)
        img_files.sort()

        for img in img_files:
            image = cv2.imread(img)
            i_name = '.'.join(img.split('.')[:-1])
            process_pipeline(image, i_name, 'jpg')

    elif ext in ['mp4', 'm4v']:
        try:
            v_stream = cv2.VideoCapture(file_pattern)
            frame_read, img = v_stream.read()
            count = 1
            while frame_read:
                if count >= v_start and count <= v_stop:
                    i_name = '{:0>{width}}'.format(count, width=7)
                    process_pipeline(img, i_name, 'j90')
                frame_read, img = v_stream.read()
                count += 1
        except:
            pass


if __name__ == "__main__":
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    modname = 'espcn'
    upsize = 4
    file_pattern = '1003_all_hi.mp4'
    homedir = os.getenv('HOME')
    workdir = '{}/Downloads/1003_ghg/'.format(homedir)
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

    ext_based_workflows()


