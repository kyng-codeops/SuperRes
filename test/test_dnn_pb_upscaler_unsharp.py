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
        fn = 'Han Shoots Greedo.mp4'
        cmd = [
            fn,
            '-s', '1',
            '-log', 'debug',
            '-par', '8', '9'
        ]
        result = dnn_pb_upscaler_unsharp.get_cli_args(cmd)
        self.assertEqual(result.file[0], cmd[0])
        self.assertEqual(result.start, cmd[2])
        self.assertIs(result.end, None)
        self.assertEqual(result.log, cmd[4])
        self.assertEqual(result.crop, ['0', '0', '0', '0'])
        self.assertEqual(result.par, [cmd[6], cmd[7]])

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
        fn = 'classroom.mp4'
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
        fn = 'classroom.mp4'
        cmd = [
            fn, '-d', 'My_Upscale',
            '-ext', 'hfyu', '-log', 'info', '-s', '10', '-e', '20',
            '-x1', '.8', '-x0', '.1'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        files = glob.glob('*.mkv')
        self.assertTrue(len(files) > 0)
    
    def test_main_args_video_ffmpg(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')

        fn = 'classroom.mp4'
        a = fn.split('.')
        fout = '{}_{}_01-{}.mp4'.format(a[0], 'X264', 10)
        
        cmd = [
            fn, '-d', 'My_Upscale', '--crop', '40', '40', '30', '30',
            '-ext', 'x264', '-log', 'info', '-e', '10'
        ]
        
        result = dnn_pb_upscaler_unsharp.main(cmd)
                
        self.assertTrue(os.path.isdir(result))
        os.chdir(result)
        self.assertTrue(os.path.isfile(fout))

    def test_main_img2video(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        cmd = ['-ext', 'mp4v', '--img-fps', '23.9762', 
                '3433.png',
                '3434.png',
                '3435.png'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        pass

    def test_main_dir_o_imgs(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        cmd = ['-ext', 'x264', '-fps', '23.976', 'dirFiles/GreatX']
        result = dnn_pb_upscaler_unsharp.main(cmd)
        pass

if __name__ == '__main__':
    unittest.main()
