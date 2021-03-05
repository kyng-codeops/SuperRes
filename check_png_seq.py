#!/usr/bin/env python3

import glob

pngs = glob.glob('*.png')
pngs.sort()

segments = []
first = pngs[0]

for i in range(0, len(pngs)-1, 1):
    f0 = int(pngs[i].split('.')[0])
    f1 = int(pngs[i+1].split('.')[0])
    if (f1 - f0) != 1:
        print('\nBROKEN Sequency between {} and {}\n'.format(f0, f1))
        end_seg = pngs[i]
        segments.append((first, end_seg))
        first = pngs[i+1]
        end_seg = ''
    if i == len(pngs)-2:
            end_seg = pngs[i+1]
            segments.append((first, end_seg))
    
print('Finished checking {} files.'.format(len(pngs)))
print(segments)