import os

# Set the PATH and PYTHON_PATH variables
from .setup import set_environment
set_environment()

def root_path(*args) :
    return os.path.abspath( os.path.join( os.path.dirname(__file__), '..', *args ) )
    