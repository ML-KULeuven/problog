from __future__ import print_function

# Load general Python modules
import sys
import os
import glob
import tempfile
import traceback
import subprocess

# Load ProbLog modules
sys.path.insert(0, os.path.abspath( os.path.join( os.path.dirname(__file__), '..' ) ) )
from problog.program import PrologString
from problog.evaluator import SemiringSymbolic, Evaluator
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
from problog.formula import LogicDAG, LogicFormula

# Which compiled knowledge format to use? (SDD or NNF)
KNOWLEDGE = NNF

# Load Python standard web-related modules (based on Python version)
import json
import mimetypes
if sys.version_info.major == 2 :
    import BaseHTTPServer
    import urlparse
    def toBytes( string ) :
        return bytes(string)
    import urllib
    url_decode = urllib.unquote_plus
else :
    import http.server as BaseHTTPServer    
    import urllib.parse as urlparse
    def toBytes( string ) :
        return bytes(string, 'UTF-8')
    url_decode = urlparse.unquote_plus

# Contains special URL paths. Initialize with @handle_url
PATHS = {}

class handle_url(object) :
    """Decorator for adding a handler for a URL.
    
    Example: to add a handler for GET /hello?name=World

    @handle_url('/hello')
    def hello(name) :
        return 200, 'text/plain', 'Hello %s!' % name
    """
    
    def __init__(self, path) :
        self.path = path
    
    def __call__(self, f) :
        PATHS[self.path] = f


def root_path(relative_path) :
    """Translate URL path to file system path."""
    return os.path.abspath( os.path.join(os.path.dirname(__file__), relative_path.lstrip('/') ) )

def model_path(*args) :
    """Translate model path to file system path."""
    return os.path.abspath( os.path.join(os.path.dirname(__file__), '../test/', *args ) )


@handle_url('/problog')
def run_problog( model ) :
    """Evaluate the given model and return the probabilities."""
    model = model[0]
    knowledge = KNOWLEDGE
    
    try :
        formula = knowledge.createFrom( PrologString(model) )
        result = formula.evaluate()
        return 200, 'application/json', json.dumps(result)
    except Exception as err :
        return process_error(err)

@handle_url('/ground')
def run_ground( model ) :
    """Ground the program given by model and return an SVG of the resulting formula."""
    model = model[0]
    knowledge = LogicDAG
    
    try :
        formula = knowledge.createFrom( PrologString(model) )
        
        handle, filename = tempfile.mkstemp('.dot')
        with open(filename, 'w') as f :
            f.write(formula.toDot())     
        result = subprocess.check_output(['dot', '-Tsvg', filename]).decode('utf-8')
        content_type = 'application/json'
        return 200, content_type, json.dumps({ 'svg' : result, 'txt' : str(formula) })
    except Exception as err :
        return process_error(err)

def process_error( err ) :
    """Take the given error raise by ProbLog and produce a meaningful error message."""
    err_type = type(err).__name__
    if err_type == 'ParseException' :
        return 400, 'text/plain', 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line )
    elif err_type == 'UnknownClause' :
        return 400, 'text/plain', 'Predicate undefined: \'%s\'.' % (err )
    elif err_type == 'PrologInstantiationError' :
        return 400, 'text/plain', 'Arithmetic operation on uninstantiated variable.' 
    elif err_type == 'UnboundProgramError' :
        return 400, 'text/plain', 'Unbounded program or program too large.'
    else :
        traceback.print_exc()
        return 400, 'text/plain', 'Unknown error: %s' % (err_type)

@handle_url('/models')    
def list_models( ) :
    
    def extract_name(f) :
        return os.path.splitext(os.path.basename(f))[0]
    
    files = map(extract_name,glob.glob(model_path('*.pl')))
    
    return 200, 'application/json', json.dumps(files)
    
    
@handle_url('/model')
def load_model( name ) :
    name = name[0]
    filename = model_path(name + '.pl')
    try :
        with open(filename) as f :
            data = f.read()
        return 200, 'text/plain', data
    except Exception :
        return 404, 'text/plain', 'File not found!'


class ProbLogHTTP(BaseHTTPServer.BaseHTTPRequestHandler) :
            
    def do_GET(self, *args, **kwdargs) :
        """Handle a GET request.
        
        Looks up the URL path in PATHS and executes the corresponding function.
        If the path does not occur in PATHS, treats the path as a filename and serves the file.
        """
        url = urlparse.urlparse( self.path )
        
        path = url.path
        
        query = urlparse.parse_qs(url.query)
        
        action = PATHS.get(path)
        if action == None :
            self.serveFile(path)
        else :
            code, datatype, data = action( **query )
            self.send_response(code)
            self.send_header("Content-type", datatype)
            self.end_headers()
            if data :
                self.wfile.write(toBytes(data))

    def serveFile(self, filename) :
        """Serve a file."""
        if filename == '/' : filename = '/index.html'
        
        filetype, encoding = mimetypes.guess_type(filename)
        filename = root_path(filename)
        try :
            with open(filename) as f :
                data = f.read()
            self.send_response(200)
            self.send_header("Content-type", filetype)
            if encoding :
                self.send_header("Content-Encoding", encoding)
            self.end_headers()
            self.wfile.write(toBytes(data))            
        except :
            self.send_response(404)
            self.end_headers()
            self.wfile.write(toBytes('File not found!'))  
    

def main(port) :
    server_address = ('', port)
    httpd = BaseHTTPServer.HTTPServer( server_address, ProbLogHTTP )
    print ('Starting server on port %s' % port)
    httpd.serve_forever()

if __name__ == '__main__' :
    if len(sys.argv) > 1 :
        port = int(sys.argv[1])
    else :
        port = 8000
    
    main(port)

