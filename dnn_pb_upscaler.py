import numpy as np
import time
import cv2
from cv2 import dnn_superres
import plaidml.keras
import imutils
import glob
import os

cv2.ocl.setUseOpenCL(True)

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

modname = 'edsr'
upsize = 3
file_pattern = 'rac*.jpg'
homedir = os.getenv('HOME')
workdir = '{}/Desktop/set2'.format(homedir)

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

timings = np.array([])

os.chdir(workdir)
img_files = glob.glob(file_pattern)
img_files.sort()
if not os.path.isdir(m_label):
    os.mkdir(m_label)

for img in img_files:

    image = cv2.imread(img)
    # Upscale the image
    t0 = time.time()
    result = sr.upsample(image)
    t1 = time.time()
    dt = t1 - t0
    timings = np.append(timings, dt)
    cv2.imwrite('{}/{}/{}_{}.jpg'.format(workdir, m_label, '.'.join(img.split('.')[:-1]), m_label),
                result,
                [int(cv2.IMWRITE_JPEG_QUALITY), 87])
    print('{}_x{}: {} took {:.{prec}f} sec | avg {:.{prec}f} sec'.
          format(prebuilt[modname], upsize,
                 img,
                 dt,
                 np.average(timings),
                 prec=2))

# cv2.imshow("Image", image)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
