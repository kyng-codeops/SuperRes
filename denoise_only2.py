#!/usr/bin/env python3
import time
import os
import sys
import cv2
from scipy.ndimage.filters import median_filter
from files_cli_ui import CommandLineUI
import multiprocessing

class NLmeans(CommandLineUI):

    def ammend_clui(self):
        self.parser.add_argument('-hL', '--h-luma', nargs=1, required=False, default=5,
            help='integer for h on luma channel denoise')
        self.parser.add_argument('-hC', '--h-croma', nargs=1, required=False, default=5,
            help='integer for h_croma of color denoise')
        return

    def ammend_ns(self):
        """
        Add custome kv pairs to the ns.kw_cus_arg dictionary where the dictionary is
        passed to the pipeline module as is, and pipeline will pass to a function--
        where the function will need custom inputs that it will unpack from ns.kw_cus_arg
        """
        # Custom kv args w defaults (for super class overloading)
        self.ns.kw_cus_arg.update({'h_luma': int(self.ns.h_luma)})
        self.ns.kw_cus_arg.update({'h_croma': int(self.ns.h_croma)})
        return

    def bgr_function(self, image, kv_args):
        self.t0 = time.time()
        self.hL = kv_args['h_luma']
        self.hC = kv_args['h_croma']
        
        # b0, g0, r0 = cv2.split(image)
        
        self.result = cv2.fastNlMeansDenoisingColored(
            image, None, self.hL, self.hC, 7, 21)
        
        # b1, g1, r1 = cv2.split(dst)
        # dst = cv2.merge([b1, g1, r0])

        self.dt = time.time() - self.t0
        return
    
def do_pipe(args, batch_arg):
    denoiser2 = NLmeans(args, 
        description='Denoising with opencv Fast-NLMeans-color',
        default_o_dir='Denoise_out', 
        default_o_ext='pn9'
        )
    # print('{}'.format(batch_arg))
    o_file = denoiser2.process_pipeline(batch_arg[0], batch_arg[1])
    # print(o_file)
    return o_file

def main(args):
    debug_mode = False
    if '--debug' in args:
        debug_mode = True

    denoiser = NLmeans(args, 
        description='Denoising with opencv Fast-NLMeans-color',
        default_o_dir='Denoise_out', 
        default_o_ext='pn9'
        )
    
    batch_args = denoiser.get_batch()

    if debug_mode:
        o_files = denoiser.debug_batch()
    else:
        n_proc = min(multiprocessing.cpu_count(), len(batch_args))
        pool = multiprocessing.Pool(n_proc)
        multi_batch = []
        for i in batch_args:
            multi_batch.append((args, i))
        o_files = pool.starmap(do_pipe, multi_batch)
        pool.close()
        pool.join()

        # reorder all mtime atime from out-of-order processing
        for out_f in o_files:
            os.utime(out_f)

    return o_files

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv.append('-h')
    main(sys.argv[1:])