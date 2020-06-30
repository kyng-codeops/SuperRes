#!/usr/bin/env python3
import os
import sys
import time
import argparse
import concurrent.futures
import cv2
from abc import ABC, abstractmethod
import collections


class CommandLineUI(ABC):
    """ Extendable prebuilt argparse UI feeding image files and custom
    parameters to user-defined image manipulation routines. Transformed
    images can be output in both lossless and highly space efficient lossy
    formats.  Lossless formats spend more time compressing and writting
    outputs than actually manipulating images so this class also pre-packages
    both a multi-threaded and single-threaded image processing methods.
    
    Usage: 
    Abtraction methods to override are marked with decorators and
    docstring code samples.
    """
    IMW_CODE = {
    'jpg': ([cv2.IMWRITE_JPEG_QUALITY, 87], 'jpg'),
    'j90': ([cv2.IMWRITE_JPEG_QUALITY, 90], 'jpg'),
    'j100': ([cv2.IMWRITE_JPEG_QUALITY, 100], 'jpg'),
    'png': ([cv2.IMWRITE_PNG_COMPRESSION, 1], 'png'),
    'pn4': ([cv2.IMWRITE_PNG_COMPRESSION, 4], 'png'),
    'pn9': ([cv2.IMWRITE_PNG_COMPRESSION, 9], 'png'),
    'wpll': ([cv2.IMWRITE_WEBP_QUALITY, 101], 'webp')
    }

    def __init__(self, args: list, **kwargs: dict):
        """ Requirements to initialize an instance:
        Required input:

        args            A list of arguments retrived from calling program
                        like main(sys.argv[1:]) also passed to argparse(args)

        Required keywords:

        description     String to describe the image manipulation
        default_o_dir   String as a default output folder (only used if -d is 
                        omitted from user args)
        default_o_ext   String that is a key entry of cls.IMW_CODE dictionary
                        (only used when -ext is omitted from user args)
        """
        self.DESC = kwargs['description']
        self.I_O_DIR = kwargs['default_o_dir']
        self.I_O_EXT = kwargs['default_o_ext']

        self.args = args

        self._reusable_default_clui()
        self.ammend_clui()
        
        self._preprocess_cli_ns()
        self.ammend_ns()

        self.o_list = collections.deque([])

    @abstractmethod
    def bgr_function(self, image: str, kv_args: dict) -> (str, time):
        """ Override this class method with your custom function
        the self.process_pipeline() class method calls
        
        The custom function must return two variables
        
        result: array as the output image
        dt: float as the delta-time taken
        
        # # Example: from denoise added params packaged in kv_args dict
        # #
        # t0 = time.time()
        # hL = kv_args['h_luma']
        # hC = kv_args['h_croma']
        # result = cv2.fastNlMeansDenoisingColored(
        #     image, None, hL, hC, 7, 21)
        # dt = time.time() - t0
        # return result, dt
        """
        t0 = time.time()
        result = image
        dt = time.time() - t0
        return result, dt
    
    def process_pipeline(self, f: str, kv_args: dict) -> str:
        """ Repetitive workflow logistics taking an image filename with
        parameters; print UI info; save the transformed image to a new
        file in a user designated output folder with user designated output
        file type; log the writen file's fully qualified outputs to a 
        thread-safe deque.
        This routine does everything except for the actual image manipulation
        which is handed off to abstractmethod bgr_function()--which is called
        by either serial_pipeline() or mthread_pipeline().
        """
        image = cv2.imread(f)
        result, dt = self.bgr_function(image, kv_args)

        workdir = os.getcwd()
        o_path = '{}/{}'.format(workdir, kv_args['o_dir'])
        f_label = f.split('.')[:-1][0]
        
        o_fqn = '{}/{}.{}'.format(
            o_path, f_label, 
            CommandLineUI.IMW_CODE[kv_args['o_ext']][1])
        was_wrtn = cv2.imwrite(
            o_fqn, result, 
            CommandLineUI.IMW_CODE[kv_args['o_ext']][0])

        if was_wrtn:
            pl_out = 'processed {:.{prec}f} sec'.format(dt, prec=2)
            print('output\t{}:\t{}'.format(f, pl_out))
            self.o_list.append(o_fqn)   # o_list is a deque for thread safety
            return o_fqn
        return False

    @abstractmethod
    def ammend_ns(self):
        """ Override this class method after overloading ammend_clui(self)
        to add custom kv pairs to the self.ns.kw_cus_arg dictionary.
        The self.bgr_function should unpack any customizing parameters as
        needed matching the cli input.

        # # Example appending built-in instance namespace
        # #
        # self.ns.kw_cus_arg.update({'h_luma': int(self.ns.h_luma)})
        # self.ns.kw_cus_arg.update({'h_croma': int(self.ns.h_croma)})
        """
        pass

    @abstractmethod
    def ammend_clui(self):
        """ Override this class method to extend argparse arguments
        sepecific to the bgr_function(self).

        # # Example overload from denoiser argparse:
        # #
        # self.parser.add_argument('-hL', '--h-luma', nargs=1, default=5,
        #     required=False, 
        #     help='integer for h on luma channel')
        # self.parser.add_argument('-hC', '--h-croma', nargs=1, default=5,
        #     required=False, 
        #     help='integer for h_croma of color denoise')
        """
        pass

    def _reusable_default_clui(self):
        """ Base common setup for argparse used repeately to ask for files vs
        beginning:ending image/frame sequence file names along with on-the-fly
        output folder renaming or output image type customizations.
        """
        self.parser = argparse.ArgumentParser(description=self.DESC)
        self.parser.add_argument('-ext', '--out-ext', nargs=1, required=False,
            default=self.I_O_EXT,
            choices=self.IMW_CODE.keys(),
            help='string to define output type')
        self.parser.add_argument('-d', '--out-dir', nargs=1, required=False, 
            default=self.I_O_DIR,
            help='string to override default output subfolder name')
        self.parser.add_argument('--debug', action='store_true', 
            help='used for special debugging logic')
        
        # Take input as file names w/wo wildcards or frame_no.ext sequence
        self.subparsers = self.parser.add_subparsers(
            help='provide files or image sequence numbers')

        self.f_parser = self.subparsers.add_parser('files')
        self.i_parser = self.subparsers.add_parser('imseq')

        self.f_parser.add_argument('file', nargs='*',
            help='file_name like MyPic.jpg or file_name_pattern IMG*.png')
        
        self.i_parser.add_argument('-b', '--begin', nargs=1, 
            help='example start with 0001.png')
        self.i_parser.add_argument('-e', '--end', nargs=1, 
            help='example end with 0010.png')

    def _preprocess_cli_ns(self):
        """ Simplify the argparse namespace self.ns to reduce coding checks on
        lists vs strings and attribute existance. 
        Requirements:  self.ns contains dictionary self.ns.kw_cus_arg
        """
        self.ns = self.parser.parse_args(self.args)

        # flatten namespace single item lists where nargs=1
        for k, v in self.ns.__dict__.items():
            if isinstance(v, list):
                if len(v) == 1:
                    self.ns.__dict__.update({k: v[0]})

        # For args with defaults to friendly kv
        self.ns.kw_cus_arg = {'o_ext': self.ns.out_ext}
        self.ns.kw_cus_arg.update({'o_dir': self.ns.out_dir})

    def setup_batch(self) -> list:
        """ Process user inputs captured after calling instance instantiation.
        It generates an interally stored batch list needed before calling one 
        of two execution processing methods; 
        serial_pipeline() -or- mthread_pipeline()
        """
        self.im_files_flg = False
        
        try:
            file_pattern = self.ns.file
            self.im_files_flg = True
        except AttributeError:
            try:
                seq_b = self.ns.begin
                seq_e = self.ns.end
                if len(seq_b) != len(seq_e):
                    raise RuntimeError(
                        'Length of chars in begining and ending inconsistant'
                        )
            except AttributeError:
                raise RuntimeError('missing inputs')
        
        # Create a subfolder for output using model label name with x factor
        if not os.path.isdir(self.ns.kw_cus_arg['o_dir']):
            os.mkdir(self.ns.kw_cus_arg['o_dir'])
        print('\nDestination folder set:\n{}/{}'.format(
                os.getcwd(),
                self.ns.kw_cus_arg['o_dir']
                )
            )

        self.batch_args = []

        if self.im_files_flg:
            for f in file_pattern:
                self.batch_args.append(
                    (f, self.ns.kw_cus_arg)
                    )
        else:
            f = seq_b.split('.')
            if os.path.isfile('{}.{}'.format(f[0], f[1])):
                ext = f[1]
                pad = len(f[0])
                frame_b = int(f[0])
                frame_e = int(seq_e.split('.')[0]) + 1
                
                for frame in range(frame_b, frame_e, 1):
                    f_name = '{:0>{width}}.{}'.format(
                        frame, ext, width=pad)
                    self.batch_args.append(
                        (f_name, self.ns.kw_cus_arg))            
        return

    def serial_pipeline(self) -> list:
        """ Call this after calling setup_batch() as one of two options.
        Debugging process logic while implementing multiprocessing can be 
        problematic. Call this to execute logic without multi-threading 
        (full multi-processing code removed due to complexity perf trade-off).
        
        Returns a list of all output files processed and written.
        """ 
        for b_arg in self.batch_args:
            f, kv_args = [tups for tups in b_arg]
            self.process_pipeline(f, kv_args)
        return self.o_list

    def mthread_pipeline(self):
        """ Multithread code used to feed and execute self.serial_pipeline()
        """
        n_proc = min(os.cpu_count(), len(self.batch_args))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
            o_files = executor.map(self._thread_function, range(len(self.batch_args)))
        
        # reorder all mtime atime from out-of-order processing
        chk_f = []
        for out_f in o_files:
            os.utime(out_f)
            chk_f.append(out_f)
        if chk_f != sorted(self.o_list):
            print('WARNING: deque(o_list) possibly mangled by race condition')
            return
        return chk_f

    def _thread_function(self, i: int) -> str:
        """ Proxy function to make threading code legible
        """
        tup = self.batch_args[i]
        return self.process_pipeline(tup[0], tup[1])

# Benchmarking m-p (w/wo class obj) and m-th using nlmeans_denoise imseq
# 
# multiprocessing non-class-non-template test ran in 13.6s
# multiprocessing class test ran in 14s
# multithreading test ran in 16s
# mp is faster but complicated to template