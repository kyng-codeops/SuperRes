#!/usr/bin/env python3
import bpy
import os
import sys
import glob

from bpy import context

""" Code notes:gist
---new directions (12/2020)---
Under "Text Editor" of Blender load the script file
Put script file, .blend file all in the relative path of the frames directors or above it
End the workdir variable below and the channel loading numbers at the very end of the script
run the script form the text editor

All modules are now self-contained so no more needing to link files in the blender addons folder

---old directions---
Google lookup on how to?
This is for Class/objects not regular function import
    import importlib.util as libu
    spec = libu.spec_from_file_location("module.name", "/path/to/file.py"))
    foo = libu.module_from_spec(spec)
    spec.loader.exec_module(foo)
    foo.MyClass()

First idea was to create a series of python functions that could be easily 
imported into the 'Blender Python Interpreter'.  In order for the Blender
internal interpreter to find modules, it must be part of Blenders Python
module search path.  One brute force method was to manually link py files into
one of folders Blender searches.
On macOS (currently mojave with Blender 2.82):
    cd /Users/kng/Applications/Blender.app/Contents/Resources/2.82/scripts/addons
    ln -s /Users/kng/Downloads/jedi/contiguous_ranges.py ./

in the blender python interpreter we just do:
    import blender_frame_loader as sw

Then to call functions to refresh the list of segments as tuples like:

    sw.png_segs = sw.rescan_frames()

To load all segments found in sw.png_segs into Blender channel 4:
    sw.load_all_segments(4)

To load just the last segment in the png_segs[] list into channel 4:
    sw.load_last_segment(4)

To load a specific index entry [3] from png_segs[] into channel 4:
    sw.load_idx_segment(4, 3)
"""

projdir = bpy.path.abspath('//')
workdir = bpy.path.abspath('//GreatX_DVD_src_1.00x_628x272_Artemis-HQ')
framext = 'tiff'

scene = context.scene
tpz_offset = 1

def get_png_ranges(workdir):
    os.chdir(workdir)
    pngs = glob.glob('*.' + framext)
    pngs.sort()

    segments = []
    first = pngs[0]

    for i in range(0, len(pngs)-1, 1):
        f0 = int(pngs[i].split('.')[0])
        f1 = int(pngs[i+1].split('.')[0])
        if (f1 - f0) != 1:
            # print('\nBROKEN Sequency between {} and {}\n'.format(f0, f1))
            end_seg = pngs[i]
            segments.append((first, end_seg))
            first = pngs[i+1]
            end_seg = ''
        if i == (len(pngs)-2):
            end_seg = pngs[i+1]
            segments.append((first, end_seg))
    
    # print('Finished checking {} files.'.format(len(pngs)))
    return segments

def rescan_frames():
    s = get_png_ranges(workdir)
    return s

def load_all_segments(chan):
    scene.sequence_editor_create()

    for seg in png_segs:
        start_prefix = seg[0].split('.')[:-1][0]
        stop_prefix = seg[1].split('.')[:-1][0]
        w = len(start_prefix)
        start_frame = int(start_prefix)
        n_frames = int(stop_prefix) - start_frame + 1
        bl_start_frame = start_frame + tpz_offset
        
        filepath = '{}/{}'.format(workdir, seg[0])
        seq = scene.sequence_editor.sequences.new_image(
                    name="new_imgs {}".format(start_frame),
                    filepath=filepath,
                    channel=chan, frame_start=bl_start_frame)
        print('starting: {} @frame {} len {}'.format(filepath, bl_start_frame, n_frames))
        for fr in range(start_frame+1, int(stop_prefix)+1, 1):
            #FIXME: png tiff hard coded fix
            img_file = '{:0>{width}}.{}'.format(fr, width=w, framext)
            seq.elements.append(img_file)
            # print('\tadding: {}'.format(filepath))
        # print('\n')
        # seq.strip.SequenceCrop(min_x = 47)

def load_last_segment(chan):
    scene.sequence_editor_create()
    seg = png_segs[-1]
    start_prefix = seg[0].split('.')[:-1][0]
    stop_prefix = seg[1].split('.')[:-1][0]
    w = len(start_prefix)
    start_frame = int(start_prefix)
    n_frames = int(stop_prefix) - start_frame + 1
    bl_start_frame = start_frame + tpz_offset
    
    filepath = '{}/{}'.format(workdir, seg[0])
    seq = scene.sequence_editor.sequences.new_image(
                name="new_imgs {}".format(start_frame),
                filepath=filepath,
                channel=chan, frame_start=bl_start_frame)
    print('starting: {} @frame {} len {}'.format(filepath, bl_start_frame, n_frames))
    for fr in range(start_frame+1, int(stop_prefix)+1, 1):
        img_file = '{:0>{width}}.png'.format(fr, width=w)
        seq.elements.append(img_file)

def load_idx_segment(chan, idx):
    scene.sequence_editor_create()
    seg = png_segs[idx]
    start_prefix = seg[0].split('.')[:-1][0]
    stop_prefix = seg[1].split('.')[:-1][0]
    w = len(start_prefix)
    start_frame = int(start_prefix)
    n_frames = int(stop_prefix) - start_frame + 1
    bl_start_frame = start_frame + tpz_offset
    
    filepath = '{}/{}'.format(workdir, seg[0])
    seq = scene.sequence_editor.sequences.new_image(
                name="new_imgs {}".format(start_frame),
                filepath=filepath,
                channel=chan, frame_start=bl_start_frame)
    print('starting: {} @frame {} len {}'.format(filepath, bl_start_frame, n_frames))
    for fr in range(start_frame+1, int(stop_prefix)+1, 1):
        img_file = '{:0>{width}}.png'.format(fr, width=w)
        seq.elements.append(img_file)


png_segs = get_png_ranges(workdir)
# To use: uncomment one of the following method calls with the correct Blender
# VSE channel as the target to load the strip into.
load_all_segments(4)
#load_last_segment(13)
#load_idx_segment(12, 1)
