#!/usr/bin/env python3

import cv2
from scipy.ndimage.filters import median_filter
import argparse
import sys
import os
import numpy as np
import time
import concurrent.futures

class CommandLineUI(object):
    
    IMW_CODE = {
    'jpg': ([cv2.IMWRITE_JPEG_QUALITY, 87], 'jpg'),
    'j90': ([cv2.IMWRITE_JPEG_QUALITY, 90], 'jpg'),
    'j100': ([cv2.IMWRITE_JPEG_QUALITY, 100], 'jpg'),
    'png': ([cv2.IMWRITE_PNG_COMPRESSION, 1], 'png'),
    'pn4': ([cv2.IMWRITE_PNG_COMPRESSION, 4], 'png'),
    'pn9': ([cv2.IMWRITE_PNG_COMPRESSION, 9], 'png'),
    'wpll': ([cv2.IMWRITE_WEBP_QUALITY, 101], 'webp')
    }

    def __init__(self, args, **kwargs):
        self.DESC = kwargs['description']
        self.I_O_DIR = kwargs['default_o_dir']
        self.I_O_EXT = kwargs['default_o_ext']

        self.args = args

        self._reusable_default_clui()
        self.ammend_clui()
        
        self._preprocess_cli_ns()
        self.ammend_ns()

        self.o_list = []

    def bgr_function(self, image, kv_args):
        """
        Overload this class method with your custom function
        the self.process_pipeline() class method calls
        
        The custom function must return two variables
        
        result: array as the output image
        dt: float as the delta-time taken
        
        # Example: from denoise added params packaged in kv_args dict
        #
        t0 = time.time()
        hL = kv_args['h_luma']
        hC = kv_args['h_croma']
        result = cv2.fastNlMeansDenoisingColored(
            image, None, hL, hC, 7, 21)
        dt = time.time() - t0
        return result, dt
        """
        t0 = time.time()
        result = image
        dt = time.time() - t0
        return result, dt
    
    def process_pipeline(self, f, kv_args):
        
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
            self.o_list.append(o_fqn)
            return o_fqn
        return False

    def ammend_ns(self):
        """
        Overload this class method after overloading ammend_clui(self)
        to add custom kv pairs to the self.ns.kw_cus_arg dict where the 
        dictionary is passed to the pipeline module as is, and pipeline will
        pass thru to the brg_function. The function  should unpack any
        customizing parameters it needs from the cli input
        
        #
        # Example appending built-in instance namespace
        #
        self.ns.kw_cus_arg.update({'h_luma': int(self.ns.h_luma)})
        self.ns.kw_cus_arg.update({'h_croma': int(self.ns.h_croma)})
        """
        return

    def ammend_clui(self):
        """
        Overload this class method to extend argparse arguments
        sepecific to the bgr_function(self).

        # Example overload from denoiser argparse:

        self.parser.add_argument('-hL', '--h-luma', nargs=1, default=5,
            required=False, 
            help='integer for h on luma channel')
        self.parser.add_argument('-hC', '--h-croma', nargs=1, default=5,
            required=False, 
            help='integer for h_croma of color denoise')
        """
        return

    def _reusable_default_clui(self):
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
        
        return

    def setup_batch(self):
        """
        ns is the namespace output of argparse
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

        # self.o_list = []
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
        
        return self.batch_args

    def serial_pipeline(self):
        # debugging multiprocess in vscode seems to still be a problem            
        for b_arg in self.batch_args:
            f, kv_args = [tups for tups in b_arg]
            self.process_pipeline(f, kv_args)
        return self.o_list

    def mthread_pipeline(self):
        # multithread version of self.serial_pipeline()
        n_proc = min(os.cpu_count(), len(self.batch_args))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
            o_files = executor.map(self._thread_function, range(len(self.batch_args)))
        # reorder all mtime atime from out-of-order processing
        self.o_list.sort()
        chk_f = []
        for out_f in o_files:
            os.utime(out_f)
            chk_f.append(out_f)
        if chk_f != self.o_list:
            print('WARNING: o_list possibly mangled by race condition')
            return
        return chk_f

    def _thread_function(self, batch_arg_i):
            tup = self.batch_args[batch_arg_i]
            return self.process_pipeline(tup[0], tup[1])

    def _preprocess_cli_ns(self):
        self.ns = self.parser.parse_args(self.args)

        # flatten namespace single item lists where nargs=1
        for self.k, self.v in self.ns.__dict__.items():
            if isinstance(self.v, list):
                if len(self.v) == 1:
                    self.ns.__dict__.update({self.k: self.v[0]})

        # For args with defaults to friendly kv
        self.ns.kw_cus_arg = {'o_ext': self.ns.out_ext}
        self.ns.kw_cus_arg.update({'o_dir': self.ns.out_dir})

        return


# Benchmarking m-p (w/wo class obj) and m-th using nlmeans_denoise imseq
# 
# multiprocessing non-class-non-template test ran in 13.6s
# multiprocessing class test ran in 14s
# multithreading test ran in 16s
# mp is faster but complicated to template
