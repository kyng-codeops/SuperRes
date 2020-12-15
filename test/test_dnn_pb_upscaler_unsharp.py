import unittest
# REM: next line only works after pyc code cached in __pycache__ of subfoler
import dnn_pb_upscaler_unsharp

import sys
import os
import glob

HOMEDIR = os.getenv('HOME')


class TestUnits(unittest.TestCase):

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
        cmd = [ fn ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        output = '{}/{}'.format(result, '001.jpg')
        self.assertFalse(os.path.isfile(output))


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
        for i in range(int(start), int(end) + 1, 1):
            output = '{}/{:0>{width}}.jpg'.format(result, i, width=7)
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
            fn, '-d', 'My_Upscale',
            '-ext', 'avc1',
            '-x1', '.8', '-x0', '.1', '-e', '8'
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
            '-ext', 'ffv1',
            '-x1', '.8', '-x0', '.1', '-e', '8'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        files = glob.glob('*.mkv')
        self.assertTrue(len(files) > 0)
    

if __name__ == '__main__':
    unittest.main()
