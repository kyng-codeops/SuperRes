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
        fn = '001.png'
        cmd = [ fn ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        output = '{}/{}'.format(result, '001.jpg')
        self.assertTrue(os.path.isfile(output))

    def test_main_args_video(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # os.chdir('test')
        fn = 'Han Shoots Greedo.mp4'
        cmd = [
            fn,
            '-s', '8',
            '-e', '16',
            '--postsharpen', '0.7'
        ]
        result = dnn_pb_upscaler_unsharp.main(cmd)
        output = '{}/{:0>{width}}.*'.format(result, int(cmd[2]), width=7)
        file = glob.glob(output)
        # check first file exists
        o_ext = file[0].split('.')[-1]
        self.assertTrue(os.path.isfile(file[0]))
        
        # check last file exists
        output = '{}/{:0>{width}}.{}'.format(result, int(cmd[4]), o_ext, width=7)
        self.assertTrue(os.path.isfile(output))


if __name__ == '__main__':
    unittest.main()
