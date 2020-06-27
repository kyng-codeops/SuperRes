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

class TestNonClassUnits(unittest.TestCase):
    """ Test original versions before OOP conversion
    """

    def test_nonclassed_denoise_imseq(self):
        """ Specify -hL with imseq -b nad -e
        default -hC -o/ext and -d
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('\nWorking in test directory')
        cmd = [ '-hL', '2', 'imseq', '-b', '0000008.png', '-e', '0000012.png' ]
        results = denoise_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
            os.remove(o_file)
        
    def test_nonclassed_denoise_files(self):
        """ Specify -hC -o/ext -d and files
        default -hL
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('\nWorking in test directory')
        cmd = [ '-hC', '2', '-o','jpg', '-d', 'NewDenoise', 
                'files', '0000010.png', '0000012.png' ]
        results = denoise_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
            os.remove(o_file)


class TestClassUnits(unittest.TestCase):
    """ Test versions of code that implement OOP
    """

    def test_obj_denoise_indi_files(self):
        """ Creating reusable interface with denoise as template
        Default -hL but override output file type and output directory
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('\nWorking in test directory')
        cmd = [ '-hc', '2', '-ext','jpg', '-d', 'NewDenoise', 'files', 
                '0000010.png', '0000012.png' ]
        results = denoise_only2.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
            os.remove(o_file)
    
    def test_obj_denoise_imseq(self):
        """ Default -hC output type and output directory but override -hL
        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('\nWorking in test directory')
        cmd = [ '--debug', '-hl', '2', 'imseq', 
                '-b', '0000008.png', '-e', '0000012.png']
        cmd = [ '-hl', '2', 'imseq', '-b', '0000008.png', '-e', '0000012.png' ]
        results = denoise_only2.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
            os.remove(o_file)


if __name__ == '__main__':
    unittest.main()