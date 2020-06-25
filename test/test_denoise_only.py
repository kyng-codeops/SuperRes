import unittest
# REM: next line only works after pyc code cached in __pycache__ of subfoler
import sys
import os
import glob
sys.path.append('../SuperRes')
import denoise_only
# import iles_cli_ui
import denoise_only2

HOMEDIR = os.getenv('HOME')

class TestUnits(unittest.TestCase):

    def test_imseq(self):
        """
        Default -hC output type and output directory but override -hL
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = ['-hL', '2', 'imseq', '-b', '0000008.png', '-e', '0000012.png']
        results = denoise_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
        
    def test_indi_files_cust_ext(self):
        """
        Default -hL but override output file type and output directory
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = [ '-hC', '2', '-o','jpg', '-d', 'NewDenoise', 'files', '0000010.png', '0000012.png' ]
        results = denoise_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))


    def test_obj_denoise_indi_files(self):
        """
        Creating reusable interface with denoise as template
        Default -hL but override output file type and output directory
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = [ '-hC', '2', '-ext','jpg', '-d', 'NewDenoise', 'files', '0000010.png', '0000012.png' ]
        results = denoise_only2.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
    
    def test_obj_imseq(self):
        """
        Default -hC output type and output directory but override -hL
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = ['--debug', '-hL', '2', 'imseq', '-b', '0000008.png', '-e', '0000012.png']
        cmd = ['-hL', '2', 'imseq', '-b', '0000008.png', '-e', '0000012.png']
        results = denoise_only2.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))


if __name__ == '__main__':
    unittest.main()