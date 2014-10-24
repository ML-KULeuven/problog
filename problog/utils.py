from __future__ import print_function

import os, tempfile, shutil

def local_path(*path) :
    """Given a path relative from the problog package root directory, returns absolute path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *path))

class TemporaryDirectory(object) :
    """Context manager for creating a temporary directory. 
    
    Use this in a with-block, or call create() and cleanup() explicitly."""
    
    NEVER_KEEP=0
    KEEP_ON_ERROR=1
    ALWAYS_KEEP=2
    
    def __init__(self, tmpdir=None, persistent=NEVER_KEEP) :
        # persistence :
        #    0: directory is always removed on exit
        #    1: directory is removed, unless exit was caused by an error
        #    2: directory is never removed
        
        self.__tmpdir = tmpdir
        self.__persistent = persistent
        
    def create(self) :
        if self.__tmpdir == None :
            self.__tmpdir = tempfile.mkdtemp()
            # self.log('Using temporary directory', self.__tmpdir, verbose=1)
        elif not os.path.exists(self.__tmpdir) :
            os.makedirs(self.__tmpdir)
        else :  # using non-temporary, existing directory => NEVER delete this
            self.__persistent = self.ALWAYS_KEEP

    def cleanup(self, error=False) :
        if self.__persistent == self.NEVER_KEEP or (self.__persistent == self.KEEP_ON_ERROR and not error) :
            shutil.rmtree(self.__tmpdir)
                
    def __enter__(self) :
        self.create()
        return self
        
    def __exit__(self, exc_type, value, traceback) :
        self.cleanup(exc_type != None)
        
    def abspath(self, *relative_filename) :
        return os.path.join(self.__tmpdir, *relative_filename)    
