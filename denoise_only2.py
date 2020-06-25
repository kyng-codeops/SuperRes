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
        """
        Custom params for nlmeans denoising with opencv
        """
        self.parser.add_argument('-hl', '--h-luma', nargs=1, required=False, default=5,
            help='integer for h on luma channel denoise')
        self.parser.add_argument('-hc', '--h-croma', nargs=1, required=False, default=5,
            help='integer for h_croma of color denoise')
        self.parser.add_argument('--mp', action='store_true', 
            help='used for special multiprocessing code')
        return

    def ammend_ns(self):
        """
        Append kv pairs to class instance namespace ns.kw_cus_arg
        """
        # Custom kv args w defaults (for super class overloading)
        self.ns.kw_cus_arg.update({'h_luma': int(self.ns.h_luma)})
        self.ns.kw_cus_arg.update({'h_croma': int(self.ns.h_croma)})
        return

    def bgr_function(self, image, kv_args):
        t0 = time.time()
        hL = kv_args['h_luma']
        hC = kv_args['h_croma']
        
        result = cv2.fastNlMeansDenoisingColored(
            image, None, hL, hC, 7, 21)

        dt = time.time() - t0
        return result, dt
    
## For multiprocessing extention not part of template
def do_pipe(args, batch_arg):
    denoiser2 = NLmeans(args, 
        description='Denoising with opencv Fast-NLMeans-color',
        default_o_dir='Denoise_out', 
        default_o_ext='pn9'
        )
    o_file = denoiser2.process_pipeline(batch_arg[0], batch_arg[1])
    return o_file

def main(args):
    debug_mode = False
    mp_mode = False
    if '--debug' in args:
        debug_mode = True
    if '--mp' in args:
        mp_mode = True

    denoiser = NLmeans(args, 
        description='Denoising with opencv Fast-NLMeans-color',
        default_o_dir='Denoise_out', 
        default_o_ext='pn9'
        )
    
    batch_args = denoiser.setup_batch()

    if debug_mode:
        o_files = denoiser.serial_pipeline()
    if mp_mode:
        # For multiprocess benchmarking agianst threads
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
    else:
        # multithreading class method
        o_files = denoiser.mthread_pipeline()

    return o_files

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv.append('-h')
    main(sys.argv[1:])