import unittest
# REM: next line only works after pyc code cached in __pycache__ of subfoler
import sys
import os
import glob
sys.path.append('../SuperRes')
import unsharp_only

HOMEDIR = os.getenv('HOME')

class TestUnits(unittest.TestCase):

    def test_imseq(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = ['-s', '1', '-m', '.7', 'imseq', '-b', '0000008.png', '-e', '0000012.png']
        results = unsharp_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
        
    def test_indi_files_cust_ext(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        # cmd = [ '-s', '1', '-m', '0.7', '-ext','jpg', 'files', '0000010.png', '0000012.png' ]
        cmd = [ '-m', '0.7', '-ext','jpg', 'files', '0000010.png', '0000012.png' ]
        results = unsharp_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))


if __name__ == '__main__':
    unittest.main()