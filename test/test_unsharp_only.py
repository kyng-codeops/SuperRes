import unittest
# REM: next line only works after pyc code cached in __pycache__ of subfoler
import sys
import os
import glob
sys.path.append('../SuperRes')
import unsharp_only

HOMEDIR = os.getenv('HOME')

class TestUnits(unittest.TestCase):

    def test_argparse_imseq(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = ['-s', '1', '-m', '.7', 'imseq', '-b', '007.png', '-e', '010.png']
        result = unsharp_only.main(cmd)
        os.chdir(result)
        self.assertTrue(os.path.isfile('007.png'))
        self.assertTrue(os.path.isfile('008.png'))
        self.assertTrue(os.path.isfile('009.png'))
        self.assertTrue(os.path.isfile('010.png'))
        pass

    def test_get_cli_args(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = [ '-s', '1', '-m', '0.7', '-o','jpg', 'files', '006.png', '007.png' ]
        result = unsharp_only.main(cmd)
        os.chdir(result)
        self.assertTrue(os.path.isfile('006.jpg'))
        self.assertTrue(os.path.isfile('007.jpg'))

if __name__ == '__main__':
    unittest.main()