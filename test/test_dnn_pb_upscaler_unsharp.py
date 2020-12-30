import unittest
# REM: next line only works after pyc code cached in __pycache__ of subfoler
import dnn_pb_upscaler_unsharp

import sys
import os
import glob

HOMEDIR = os.getenv('HOME')


class TestUnits(unittest.TestCase):

    def test_main_args_null(self):
        # should just see the help menu then argparse will do SystemExit
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        cmd = []
        try:
            result = dnn_pb_upscaler_unsharp.main(cmd)
        except SystemExit:
            pass    
        try:
            self.assertFalse(isinstance(result, str))
        except UnboundLocalError:
            pass

    def test_get_cli_args(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        fn = 'test_vid.mkv'
        fn = 'Han Shoots Greedo.mp4'
        cmd = [
            fn,
            '-s', '1',
            '-e', '10'
        ]
        result = dnn_pb_upscaler_unsharp.get_cli_args(cmd)
        self.assertEqual(result.file[0], cmd[0])
        self.assertEqual(result.start, cmd[2])
        self.assertEqual(result.end, cmd[4])

    def test_main_args_image(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = 'none.png'
        # fn = '41.png'
        cmd = [ fn ]
        try:
            result = dnn_pb_upscaler_unsharp.main(cmd)
        except SystemExit:
            pass
        try:
            self.assertFalse(isinstance(result, str))
        except UnboundLocalError:
            pass


class TestVideoExtraction(unittest.TestCase):
    
    def test_main_args_video_range(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = 'Han Shoots Greedo.mp4'
        start = '8'
        end = '12'
        cmd = [
            fn, '-d', 'My_Upscale',
            '-s', start,
            '-e', end, '-ext', 'jpg',
            '-x1', '.8', '-x0', '0'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        files = []
        wpad = len(str(end))
        for i in range(int(start), int(end) + 1, 1):
            output = '{}/{:0>{width}}.jpg'.format(result, i, width=wpad)
            self.assertTrue(os.path.isfile(output))
    
    def test_main_args_video_end(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = 'Han Shoots Greedo.mp4'
        cmd = [
            fn, '-d', 'My_Upscale',
            '-ext', 'png',
            '-x1', '.8', '-x0', '.1', '-e', '8'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        files = glob.glob('*.png')
        self.assertTrue(len(files) > 0)

    def test_main_args_video_out1(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = 'Han Shoots Greedo.mp4'
        cmd = [
            fn, '-d', 'My_Upscale', '-log', 'info',
            '-ext', 'mp4v', '-ese'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        files = glob.glob('*.mp4')
        self.assertTrue(len(files) > 0)
        
    def test_main_args_video_out2(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = 'Han Shoots Greedo.mp4'
        cmd = [
            fn, '-d', 'My_Upscale',
            '-ext', 'hfyu', '-log', 'info', '-s', '3600',
            '-x1', '.8', '-x0', '.1'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        files = glob.glob('*.mkv')
        self.assertTrue(len(files) > 0)
    
    def test_main_args_video_sr10(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = 'Han Shoots Greedo.mp4'
        cmd = [
            fn, '-d', 'My_Upscale', '--crop', '10', '10', '30', '30',
            '-ext', 'hev1', '-log', 'info', '-s', '3600'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        files = glob.glob('*.mkv')
        self.assertTrue(len(files) > 0)


if __name__ == '__main__':
    unittest.main()
