#!/usr/bin/env python3
import glob
import os

def get_png_ranges(run_dir):
    os.chdir(run_dir)
    pngs = glob.glob('*.png')
    pngs.sort()

    segments = []
    init_seg = ''

    n_png_files = len(pngs)

    try:
        for i in range(0, n_png_files, 1):
            if init_seg == '':
                init_seg = pngs[i]  # initalize trap segment beginning
            chk_current = pngs[i]
            f0 = int(chk_current.split('.')[0]) # int version of filename
            try:
                f1 = int(pngs[i+1].split('.')[0])   # int version of next filename
                if (f1 - f0) != 1:
                    segments.append((init_seg, chk_current))  # store non-sequencial break-point
                    init_seg = ''   # reset trap for next segment beginning
            except IndexError:
                pass    # no more next files to load as f1

        segments.append((init_seg, chk_current))    # last continuous segment becomes last attempted check
        for i in range(0, len(segments), 1):
            print(i, segments[i])

    except UnboundLocalError:
        if len(pngs) < 1:
            raise('\n*** Problem: No png files found in current directoery! ***\n')

    print('Finished checking {} files.'.format(len(pngs)))

    return segments

if __name__ == '__main__':
    run_dir = os.getcwd()
    get_png_ranges(run_dir)
