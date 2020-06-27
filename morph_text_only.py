#!/usr/bin/env python3
import time
import sys
import cv2
import numpy as np
from files_cli_ui import CommandLineUI


class MorphEX(CommandLineUI):

    def ammend_ns(self):
        pass

    def ammend_clui(self):
        pass

    def bgr_function(self, image, kv_args):
        t0 = time.time()
        
        # FIXME: doesn't work maybe because image needs to be grayscale first 
        # ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
        # result = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, k)

        k = np.ones((3,3),np.uint8)
        result = cv2.erode(image, k, iterations=1)

        # k = np.ones((5,5),np.uint8)
        # result = cv2.morphologyEx(image, cv2.MORPH_OPEN, k)

        dt = time.time() - t0
        return result, dt


DESC = 'Letter fix with opencv MorphologyEX'
O_DIR = 'LetterFix'
O_EXT = 'pn9'

def main(args):
    debug_mode = False
    if '--debug' in args:
        debug_mode = True

    txt_morph = MorphEX(args, 
        description=DESC,
        default_o_dir=O_DIR, 
        default_o_ext=O_EXT
        )
    
    txt_morph.setup_batch()

    if debug_mode:
        o_files = txt_morph.serial_pipeline()
    else:
        o_files = txt_morph.mthread_pipeline()

    return o_files

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv.append('-h')
    main(sys.argv[1:])