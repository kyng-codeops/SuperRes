import unittest
import sys
import os
import glob
# import py_compile
# py_compile.compile('../unsharp_only.py')
# or use python -m compileall .
import unsharp_only

HOMEDIR = os.getenv('HOME')

class TestUnits(unittest.TestCase):

    def test_class_help_init(self):
        """ call help() on class to view docstrings and test object instantiates
        """
        help(unsharp_only.Unsharp)
        sharpen = unsharp_only.Unsharp([], description='', 
                    default_o_dir='', 
                    default_o_ext='')
        self.assert_(sharpen)
        
    def test_unsharp_imseq(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        cmd = ['-s', '1', '-m', '.7', 'imseq', '-b', '0000008.png', '-e', '0000012.png']
        results = unsharp_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
            os.remove(o_file)
        
    def test_unsharp_files_cust_ext(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        print('Working in test directory')
        # cmd = [ '-s', '1', '-m', '0.7', '-ext','jpg', 'files', '0000010.png', '0000012.png' ]
        cmd = [ '-m', '0.7', '-ext','jpg', 'files', '0000010.png', '0000012.png' ]
        results = unsharp_only.main(cmd)
        for o_file in results:
            self.assertTrue(os.path.isfile(o_file))
            os.remove(o_file)


if __name__ == '__main__':
    unittest.main()