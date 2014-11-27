
from .program import PrologString, PrologFile

from .interface import ground
from .evaluator import Evaluator


# Set the PATH and PYTHON_PATH variables
from .setup import set_environment
set_environment()