#!/usr/bin/env python3
# import argparse
import time
import sys
import os
import cv2
import numpy as np
from scipy.ndimage.filters import median_filter

from files_cli_ui import CommandLineUI


class Unsharp(CommandLineUI):

    def sharpen_channel(self, image, sigma, strength):
        t0 = time.time()

        mf_image = median_filter(image, sigma)
        lap = cv2.Laplacian(mf_image, cv2.CV_64F)
        result = image - strength * lap
        result[result > 255] = 255
        result[result < 0] = 0

        t1 = time.time()
        dt = t1 - t0
        return result, dt

    def bgr_function(self, image, kv_args):
        sigma = kv_args['sigma']
        strength = kv_args['strength']
        r1 = np.zeros_like(image)
        dt2 = 0
        for i in range(3):
            r1[:, :, i], dt = self.sharpen_channel(image[:, :, i], sigma, strength)
            dt2 = dt2 + dt
        
        result = r1
        return result, dt2
    
    def ammend_ns(self):
        """
        Overload this class method after overloading ammend_clui(self)
        to add custom kv pairs to the self.ns.kw_cus_arg dictionary.
        The self.bgr_function should unpack any customizing parameters as
        needed matching the cli input.
        """
        self.ns.kw_cus_arg.update({'sigma': int(self.ns.sigma)})
        self.ns.kw_cus_arg.update({'strength': float(self.ns.strength)})
        return

    def ammend_clui(self):
        """
        Overload this class method to extend argparse arguments
        sepecific to the bgr_function(self).
        """
        self.parser.add_argument('-s', '--sigma', nargs=1, default=1,
            help='integer for sigma of sharpening')
        self.parser.add_argument('-m', '--strength', nargs=1, default=0.7,
            help='float for strength of sharpening .7 is very light 10 too much')
    
        return


DESC = 'Unsharp mask sharpening with opencv Laplacian'
O_DIR = 'Unsharp_out'
O_EXT = 'pn9'

def main(args):
    debug_mode = False
    if '--debug' in args:
        debug_mode = True

    sharpener = Unsharp(args, 
        description=DESC,
        default_o_dir=O_DIR, 
        default_o_ext=O_EXT
        )
    
    sharpener.setup_batch()

    if debug_mode:
        o_files = sharpener.serial_pipeline()
    else:
        o_files = sharpener.mthread_pipeline()

    return o_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append('-h')
    main(sys.argv[1:])