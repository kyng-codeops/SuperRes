#!/usr/bin/env python3

import cv2
from scipy.ndimage.filters import median_filter
import argparse
import sys
import os
import numpy as np
import time
import multiprocessing

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
        self.I_O_DIR = kwargs['default_o_dir']
        self.I_O_EXT = kwargs['default_o_ext']
        self.DESC = kwargs['description']
        self.args = args

        self.reusable_default_clui()
        self.ammend_clui()
        self.preprocess_cli_ns()
        self.ammend_ns()


    def bgr_function(self, image, kv_args):
        """
        The custome function must assign two variables
            self.result as the output image and 
            self.dt as the delta-time taken
        
        #
        # Example from denoise
        #
        self.t0 = time.time()
        self.hL = kv_args['h_luma']
        self.hC = kv_args['h_croma']
        self.result = cv2.fastNlMeansDenoisingColored(
            image, None, self.hL, self.hC, 7, 21)
        self.dt = time.time() - self.t0
        """
        self.t0 = time.time()
        self.result = image
        self.dt = time.time() - self.t0
        return


    def process_pipeline(self, f, kv_args):
        
        self.image = cv2.imread(f)
        self.bgr_function(self.image, kv_args)

        self.workdir = os.getcwd()
        self.o_path = '{}/{}'.format(self.workdir, kv_args['o_dir'])
        self.f_label = f.split('.')[:-1][0]
        
        self.o_fqn = '{}/{}.{}'.format(
            self.o_path, self.f_label, self.IMW_CODE[kv_args['o_ext']][1])
        self.was_wrtn = cv2.imwrite(
            self.o_fqn, self.result, self.IMW_CODE[kv_args['o_ext']][0])

        if self.was_wrtn:
            self.pl_out = 'processed {:.{prec}f} sec'.format(self.dt, prec=2)
            print('output\t{}:\t{}'.format(f, self.pl_out))
            return self.o_fqn
        return False

    def ammend_ns(self):
        """
        Add custome kv pairs to the ns.kw_cus_arg dictionary where the 
        dictionary is passed to the pipeline module as is, and pipeline will 
        pass to a function--where the function will need custom inputs that it
        will unpack from ns.kw_cus_arg
        
        #
        # Example custom kv args w defaults (for super class overloading)
        #
        self.ns.kw_cus_arg.update({'h_luma': int(self.ns.h_luma)})
        self.ns.kw_cus_arg.update({'h_croma': int(self.ns.h_croma)})
        """
        return

    def preprocess_cli_ns(self):
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

    def ammend_clui(self):
        """
        # Example overload from denoiser argparse:

        self.parser.add_argument('-hL', '--h-luma', nargs=1, default=5,
            required=False, 
            help='integer for h on luma channel')
        self.parser.add_argument('-hC', '--h-croma', nargs=1, default=5,
            required=False, 
            help='integer for h_croma of color denoise')
        """
        return

    def reusable_default_clui(self):
        self.parser = argparse.ArgumentParser(description=self.DESC)
        self.parser.add_argument('-ext', '--out-ext', nargs=1, required=False,
            default=self.I_O_EXT,
            choices=self.IMW_CODE.keys(),
            help='string to define output type')
        self.parser.add_argument('-d', '--out-dir', nargs=1, required=False, 
            default=self.I_O_DIR,
            help='string to override default output subfolder name')
        self.parser.add_argument('--debug', action='store_true', 
            help='use single process vs multiprocessing')
        
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
        
    def get_batch(self):
        """
        ns is the namespace output of argparse
        """
        # self.ns = self.ns
        self.im_files_flg = False
        
        try:
            self.file_pattern = self.ns.file
            self.im_files_flg = True
        except AttributeError:
            self.seq_b = self.ns.begin
            self.seq_e = self.ns.end
            if len(self.seq_b) != len(self.seq_e):
                raise RuntimeError(
                    'Length of chars in begining and ending inconsistant'
                    )
        
        # Create a subfolder for output using model label name with x factor
        if not os.path.isdir(self.ns.kw_cus_arg['o_dir']):
            os.mkdir(self.ns.kw_cus_arg['o_dir'])
        print('\nDestination folder set:\n{}/{}'.format(
                os.getcwd(),
                self.ns.kw_cus_arg['o_dir']
                )
            )

        self.o_list = []
        self.batch_args = []

        if self.im_files_flg:
            for self.f in self.file_pattern:
                self.batch_args.append(
                    (self.f, self.ns.kw_cus_arg)
                    )
        else:
            self.f = self.seq_b.split('.')
            if os.path.isfile('{}.{}'.format(self.f[0], self.f[1])):
                self.ext = self.f[1]
                self.pad = len(self.f[0])
                self.frame_b = int(self.f[0])
                self.frame_e = int(self.seq_e.split('.')[0]) + 1
                
                for self.frame in range(self.frame_b, self.frame_e, 1):
                    self.f_name = '{:0>{width}}.{}'.format(
                        self.frame, self.ext, width=self.pad)
                    self.batch_args.append(
                        (self.f_name, self.ns.kw_cus_arg))            
        
        return self.batch_args

    def debug_batch(self):
        # debugging multiprocess in vscode seems to still be a problem            
        for self.b_arg in self.batch_args:
            self.f, self.kv_args = [self.tups for self.tups in self.b_arg]
            self.ofl = self.process_pipeline(self.f, self.kv_args)
            self.o_list.append(self.ofl)
        return self.o_list