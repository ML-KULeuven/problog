import os

# Set the PATH and PYTHON_PATH variables
from .setup import set_environment, gather_info
set_environment()

system_info = gather_info()

def root_path(*args) :
    return os.path.abspath( os.path.join( os.path.dirname(__file__), '..', *args ) )

# Load all submodules. This has two reasons:
#   - initializes all transformations (@transform)
#   - makes it possible to just import 'problog' and then use 
#       something like problog.program.PrologFile
from . import cnf_formula    
from . import core
from . import engine
from . import evaluator
from . import formula
from . import logic
from . import nnf_formula
from . import parser
from . import program
from . import sdd_formula
from . import util
