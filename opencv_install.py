"""
Installing opencv

First install prereqs with homebrew
DON'T install python3 using homebrew

To install these prerequisites for OpenCV on macOS execute the following
commands:

    $ brew install cmake pkg-config
    $ brew install jpeg libpng libtiff openexr
    $ brew install eigen tbb
    $ brew install ffmpeg

Optional and convenient:
    $ brew install webpquicklook
    $ brew install wget 

Use pyenv instead to install python (since as of 5/2020 3.6.5_1 brew install
has a broken pip openssl certificate issue).  Also pyenv can do multiple
versions: handy for conflicts like Keras is py 3.6 only while Tensorflow
supports 3.7.

By default pyenv does not install python library files for building opencv
from source, install of the regular.

## WARNING:
#   $ pyenv install 3.6.10' does not include the necessary framework libs
#   there is a log of bugs and hardcoded paths that make the option
#   --enable-framework=[dir]
#   fail, instead USE:

    $ env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.10


# WARNING: manual building python outside of pyenv then linking under pyenv
# has other complications (invalid suggestion).
# in virtual sub-envs for python3 (use above pyenv install instead)
# Invalid suggestion details NOT used (wasted 12+ hours with mostly
# non-functioning build):
        $ wget http://www.python.org/ftp/python/3.6.10/Python-3.6.10.tgz
        $ tar -zvxf Python-3.6.10.tgz
        $ cd Python-3.6.10
        $ mkdir $(pyenv root)/versions/3.6.10
        $ ./configure --enable-framework=$(pyenv root)/versions/3.6.10/
        $ make
        $ make install
        $ cd $(pyenv root)/versions/3.6.10
        $ ln -s Python.framework/Versions/3.6.10/bin ./bin
        $ ln -s Python.framework/Versions/3.6.10/include ./include
        $ ln -s Python.framework/Versions/3.6.10/lib ./lib
        $ ln -s Python.framework/Versions/3.6.10/share ./share

NOTE: I had the intel OpenVINO cpu optimizers for OpenCV installed using the
brew installed python3 which needs complete reinstall as the user of the pyenv
(since ~/.pyenv requires the users pyenv and brew is system wide).
So reinstall, re-optimize, everythin with OpenVINO before doing the opencv
cmake and build otherwise python3 paths will be wrong!!

Build opencv binding into the pyenv environment!!

Create a virtualenv with the new pyenv versions where interpreters are located
under ~/.pyenv/

In the virtualenv running the selected python3 interpreter (because pyenv will
have linked the right pyenv paths):

    $ pip install numpy plaidml plaidml-keras tensorflow keras

    # NOTE: do NOT install the pip opencv-contrib-python (it's a pre-compiled
    # competing ver of opencv)

    # opencv complimentary lib nice to have in the virtualevn (custom
    # contribution by Adrian)
    $ pip install imutils

# the -j4 means 4 core processor
make -j4
sudo make install

# Final step... link the library into the virtualenv site-
cd .pyenv/versions/py36cv4/lib/python3.6/site-packages/
ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-darwin.so ./cv2.so

Trying to re-compile opencv with NVIDIA CUDA and CUDNN required older version
of xcode.

NVIDIA directions:
https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html
Apple Dev with Xcode 10.2
https://developer.apple.com/download/more/

    $ sudo xcode-select -s '/Applications/Xcode 10.2.app/Contents/Developer'
    $ sudo xcode-select --install
    $ cc --version

Trying to fix OpenCL include not found (either force 3rdparty or leave unfound
Trying to fix OpenGL, brew install gtkglext (on Linux the gtk+ gl extensions &
headers are libgtkglext and libgtkglext-dev). These are were not found with
brew search, only gtkglext exist. The actual cmake complaint is only that
OpenGL GUI is not supported (meaning use ogl for cv2.imshow?)--not a big deal.

"""

import os
import sys
import urllib.request
import zipfile
import subprocess
import distutils.sysconfig


def run_realtime(cmd):
    proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=False)
    while True:
        try:
            output = proc.stdout.readline()
            print(output.decode('utf-8').strip())
            if proc.poll() is not None:
                break
        except:
            raise
    return proc.poll()


if __name__ == "__main__":

    OPENCV_VER = '4.3.0'
    HOMEDIR = os.getenv('HOME')
    VER_DIR = '{}/opencv-{}'.format(HOMEDIR, OPENCV_VER)
    VER_CON_DIR = '{}/opencv_contrib-{}'.format(HOMEDIR, OPENCV_VER)


    # Python3 Include path
    py3inc = distutils.sysconfig.get_python_inc()

    # py3bin = '{}/bin/python'.format(os.getenv('VIRTUALENV'))  
    # above is not! reliable # also VIRTUAL_ENV, PYENV_VIRTUAL_ENV
    py3bin = subprocess.check_output('which python3', shell=True)
    py3bin = py3bin.decode('utf-8').strip()

    # Get the Python3 Library path
    s = subprocess.check_output("python3-config --configdir", shell=True)
    s = s.decode("utf-8").strip()
    (M, m) = sys.version_info[:2]
    # py3lib_o = "{}/libpython{}.{}.dylib".format(s, M, m)

    py3libDir = distutils.sysconfig.get_config_var('LIBDIR')
    py3lib = '{}/{}'.format(
        py3libDir,
        distutils.sysconfig.get_config_var('LDLIBRARY'))

    py3numpy = '{}/lib/python{}.{}/site-packages/numpy/core/include/'.\
        format('/'.join(py3bin.split('/')[:-2]), M, m)
    # py3numpy = '{}/site-package/numpy/core/include'.\
    # format(distutils.sysconfig.get_config_var('BINLIBDEST')) # not in envs

    # Check path existence and prereqs
    if not os.path.isfile(py3lib):
        print('invalid py3lib={}'.format(py3lib))
    if not os.path.isdir(py3libDir):
        print('invalid py3libDir={}'.format(py3libDir))
    if not os.path.isdir(py3inc):
        print('invalid py3inc={}'.format(py3inc))
    if not os.path.isdir(py3numpy):
        print('invalid py3numpy={}'.format(py3numpy))

    brew_list = [
        'cmake', 'pkg-config',
        'jpeg', 'libpng', 'libtiff',
        'openexr', 'eigen', 'tbb', 'ffmpeg']
    brew_missing = []
    for mod in brew_list:
        # Assuming macOS user is using Homebrew for package management
        if not os.path.isdir('/usr/local/Cellar/{}'.format(mod)):
            brew_missing.append(mod)
    if len(brew_missing) > 0:
        print('You are missing some prerequisits!! Run the following:')
        print('brew install {}'.format(' '.join(brew_missing)))
        raise Exception('You are missing some prerequisits!! Run the following:\n\
                        brew install {}'.format(' '.join(brew_missing)))

    os.chdir(HOMEDIR)

    if not os.path.isdir(VER_DIR):

        url = 'https://github.com/opencv/opencv/archive/{}.zip'.\
            format(OPENCV_VER)
        zip = './opencv.zip'
        if not os.path.isfile(zip):
            print('downloading... {} from {}'.format(zip, url))
            urllib.request.urlretrieve(url, zip)
        with zipfile.ZipFile(zip, 'r') as zip_ref:
            zip_ref.extractall('./')

    if not os.path.isdir(VER_CON_DIR):
        zip = './opencv_contrib.zip'
        url = 'https://github.com/opencv/opencv_contrib/archive/{}.zip'.\
            format(OPENCV_VER)
        if not os.path.isfile(zip):
            print('downloading... {} from {}'.format(zip, url))
            urllib.request.urlretrieve(url, zip)
        with zipfile.ZipFile(zip, 'r') as zip_ref:
            zip_ref.extractall('./')

    build_dir = '{}/src/build/opencv-{}'.format(HOMEDIR, OPENCV_VER)
    if os.path.isdir(VER_DIR) and os.path.isdir(VER_CON_DIR):
        if not os.path.isdir(build_dir):
            s_dirs = build_dir.split('/')
            for sd in s_dirs[3:]:
                if not os.path.isdir(sd):
                    os.mkdir(sd)
                    os.chdir(sd)
            # if not os.path.isdir('{}/src'.format(home)):
            #     os.mkdir('{}/src'.format(home))
            # os.mkdir(build_dir)
        os.chdir(build_dir)
        d = os.getcwd()
        cmd = [
            'cmake', '-D',
            'CMAKE_BUILD_TYPE=RELEASE',
            '-D', 'CMAKE_INSTALL_PREFIX=/usr/local',
            '-D', 'OPENCV_EXTRA_MODULES_PATH={}/modules'.format(VER_CON_DIR),
            '-D', 'opencv_dnn_superres=ON',
            '-D', 'PYTHON3_LIBRARY={}'.format(py3lib),
            '-D', 'PYTHON3_INCLUDE_DIR={}'.format(py3inc),
            '-D', 'PYTHON3_EXECUTABLE={}'.format(py3bin),
            '-D', 'PYTHON3_NUMPY_INCLUDE_DIRS={}'.format(py3numpy),
            '-D', 'BUILD_opencv_python2=OFF',
            '-D', 'BUILD_opencv_python3=ON',
            '-D', 'INSTALL_PYTHON_EXAMPLES=ON',
            '-D', 'INSTALL_C_EXAMPLES=OFF',
            '-D', 'OPENCV_ENABLE_NONFREE=ON',
            '-D', 'BUILD_EXAMPLES=ON'
        ]
        cmd_ocl_normal = [
            '-D', 'WITH_OPENCL=ON',
            '-D', 'OPENCL_INCLUDE_DIRS={}/3rdparty/include'.format(VER_DIR),
            '-D', 'WITH_OPENGL=ON'
        ]

        cmd_ocl_staticlib = [
            '-D', 'WITH_OPENCL=OFF',
            '-D', 'HAVE_OPENCL=ON',
            '-D', 'HAVE_OPENCL_STATIC=ON',
            '-D', 'OPENCL_LIBRARIES=/System/Library/Frameworks/OpenCL,framework/OpenCL',
            '-D', 'OPENCL_INCLUDE_DIRS={}/3rdparty/include'.format(VER_DIR),
            '-D', 'WITH_OPENGL=ON'
        ]
        #    '-D', 'OPENCL_LIBRARIES=/path/libOpenCL.so',

        cmd_gpu_nvidia = [
            '-D', 'WITH_CUDA=ON',
            '-D', 'WITH_CUDNN=ON',
            '-D', 'OPENCV_DNN_CUDA=ON',
            '-D', 'ENABLE_FAST_MATH=1',
            '-D', 'CUDA_FAST_MATH=1',
            '-D', 'CUDA_ARCH_BIN=3.0',
            '-D', 'WITH_CUBLAS=1',
            '-D', 'CUDNN_INCLUDE_DIR=/Developer/NVIDIA/CUDA-10.1/include',
            '-D', 'CUDNN_LIBRARY=/Developer/NVIDIA/CUDA-10.1/lib/libcudnn.dylib'
        ]
        cmd = cmd + cmd_ocl_normal
        cmd.append(VER_DIR)
        print('About to run...\n\n{}\n\n'.format(' '.join(cmd)))

        #
        # Ask to run cmake
        #
        a = input(
            '\nbuild_dir: [{}]\n\nConfigure make files with "cmake"? [y/N] '.
            format(build_dir))
        if a[:1].lower() == 'y':
            # output = subprocess.check_output(cmd, universal_newlines=True, shell=False)
            # print('\n{}'.format(output.strip()))
            run_realtime(cmd)
    else:
        print(
            'Directories [{}] and [{}] do not exist.. nothing to do'.
            format(VER_DIR, VER_CON_DIR))

    #
    # Ask to run make to compile/build
    #
    a = input('\nRun "make" to compile opencv? [y/N]? ')
    if a[:1].lower() == 'y':
        a = input(
            '\n{} cpu cores detected, enter number of make threads: '.
            format(os.cpu_count())
            )
        try:
            os.chdir(build_dir)
            cmd = ['make', '-j{}'.format(int(a))]
            run_realtime(cmd)
        except ValueError:
            raise('[{}] is an invalid number'.format(a))

    #
    # Ask to install opencv build into /usr/local
    #
    a = input('\nRun "make install" to install opencv libraries to /usr/local? [y/N] ')
    if a[:1].lower() == 'y':
        cmd = ['sudo', 'make', 'install']
        run_realtime(cmd)

    #
    # Ask to link cv2.so file
    #
    src_lib_path = '/usr/local/lib/python{}.{}/site-packages/cv2/python-{}.{}'.\
        format(M, m, M, m)
    src_lib = 'cv2.cpython-{}{}m-darwin.so'.format(M, m)
    venv_root = '/'.join(py3bin.split('/')[:-2])
    venv_lib = '{}/lib/python{}.{}/site-packages/cv2.so'.format(venv_root, M, m)
    print('\nPython venv is current:\n{}'.format(venv_root))

    a = input('\nLink opencv libs to current python env? [y/N] ')
    if a[:1].lower() == 'y':
        s = '/'.join([src_lib_path, src_lib])
        if os.path.isfile(s):
            if not os.path.isfile(venv_lib):
                os.symlink(s, venv_lib)
                print('Sucessfully link: {}'.format(venv_lib))
            else:
                print('\n\tNot linking: {} already exists\n'.format(venv_lib))
