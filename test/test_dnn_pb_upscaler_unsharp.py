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
        cmd = [
            fn,
            '-s', '5',
            '-e', '10'
        ]
        result = dnn_pb_upscaler_unsharp.get_cli_args(cmd)
        self.assertEqual(result.file[0], cmd[0])
        self.assertEqual(result.start[0], cmd[2])
        self.assertEqual(result.end[0], cmd[4])

    def test_main_args_image(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = glob.glob('*.png')
        cmd = [ '-x0', '.2', '-ext', 'jpg' ] + fn
        result = dnn_pb_upscaler_unsharp.main(cmd)
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        files = glob.glob('*.jpg')
        self.assertTrue(len(files) > 0)


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


if __name__ == '__main__':
    unittest.main()
