import web
import json
import sys, os, time, shutil
import codecs
import traceback

urls = ( '/ground', 'Ground',
         '/models', 'GetModel',
         '/submit', 'Submit',
         '/docs/(.*)', 'Docs',
         '/(.*)', 'File')


                  

# Assumes mngzn can be found in PYTHONPATH or in ..
sys.path.append('..')

from problog.logic import Term, Var, Constant
from problog.logic.program import PrologString, ClauseDB
from problog.logic.engine import Debugger
from problog.logic.eb_engine import EventBasedEngine, addBuiltins
from problog.logic.formula import LogicFormula

import subprocess as sp

from main import ground

import glob, os

def accept_file(name) :
    
    allowed_extensions = frozenset([ '.html', '.css', '.js', '.dat', '.csv', '.db', '.mzn', '.arff', '.pl' ])
    
    rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    
    fullname = os.path.abspath(name)
    
    # File must be below root directory of webserver.
    if not fullname.startswith(rootdir) :
        return False
    
    # Only allow access to *.html *.js *.css
    name, ext = os.path.splitext(fullname)

    if ext in allowed_extensions :
        return True
    return False

            
class File(object) :
    
    def GET(self, file=None, **kwdargs) :
        if not file :
            file = 'index.html'
        
        if accept_file(file) :        
            with open(file) as f:
                return f.read()
        else :
            raise web.notfound()

class Docs(object) :
    
    def GET(self, file=None, **kwdargs) :
        if not file :
            file = 'index.html'
        
        path = os.path.abspath(os.path.join('../docs/build/html/', file ))
        with open(path) as f:
            return f.read()


class Ground(object) :
    
    def POST(self, **kwdargs) :
        model = web.data()

        try :
            lp = PrologString(model)    
            
            gp = ground(lp)
            gp = gp.makeAcyclic()    
            filename = '/tmp/pl.dot'
            with open(filename, 'w') as f :
                f.write(gp.toDot())            
            output = sp.check_output(['dot', '-Tsvg', filename])
            return '<strong>' + output + '</strong>'
        except Exception as err :
            s = '<strong>' + type(err).__name__ + ': ' + str(err) + '</strong><br/>'
            s += traceback.format_exc().replace('\n','<br/>').replace(' ','&nbsp;')
            return s
            
class Submit(object) :
    
    def POST(self, **kwdargs) :
        name = web.input()['name']
        data = web.input()['data']

        i = 0
        name = name.replace('.','_').replace('/','_')
        partial_name = name
        
        fullname = os.path.join(os.path.dirname(__file__), 'data/bug/', name) + '.pl'
        while os.path.exists(fullname) :
            i += 1
            partial_name = name +  '_' + str(i) 
            fullname = os.path.join(os.path.dirname(__file__), 'data/bug/', partial_name)+ '.pl'                 
        with open(fullname, 'w') as f :
            f.write(data)
        return 'bug/' + partial_name


class GetModel(object) :
    
    def POST(self, **kwdargs) :
        print 'INPUT', web.input()
        print 'DATA', web.data()
        return ''
    
    def GET(self, **kwdargs) :
        name = web.input().get('name')
        
        if name == None :
            result = '<option value="">-- select a model --</option>'
            files = [ os.path.basename(x)[:-3] for x in glob.glob('data/*.pl') ]
            files += [ 'bug/' + os.path.basename(x)[:-3] for x in glob.glob('data/bug/*.pl') ]
            for name in files :
                result += '<option value="' + name + '">' + name + '</option>'
            return result
        else :
            return self.get_model(name)
        
    def get_model(self, name) :
        file = 'data/' + name + '.pl'
        if accept_file(file) :
            with open(file) as f :
                return f.read()
        else :
            raise web.notfound()

    
        




class GenericService(object) :
    
    def POST(self, **kwdargs) :
        print 'INPUT', web.input()
        print 'DATA', web.data()
        return ''
    
    def GET(self, **kwdargs) :
        if VERBOSE :
            print 'GET', web.input()
        if 'callback' in web.input():
            # this is a jsonP call
            callback = web.input()['callback']
            result = callback+"(" + json.dumps(self.run(**web.input())) + ");"
        else:
            result = str(self.run(**web.input()))
        
        if VERBOSE : 
            print 'RESULT:', result
        return result
        
    def run(self, **kwdargs) :
        raise NotImplementedError('This is an abstract function.')

        
if __name__ == '__main__' :
    app = web.application(urls, globals(), web.profiler)
    app.run()
