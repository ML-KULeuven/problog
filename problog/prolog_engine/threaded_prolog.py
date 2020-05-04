import dill as pickle
from multiprocessing import Process, Pipe
import os
from my_pyswip import Prolog
from swip import parse_result


def load_external_file(filename, prolog, database=None):
    # Weird need to give this prolog to swi_prolog_export
    from problog.prolog_engine.extern import swi_problog_export
    import imp
    swi_problog_export.prolog = prolog
    swi_problog_export.database = database

    module_name = os.path.splitext(os.path.split(filename)[-1])[0]
    with open(filename, 'r') as extfile:
        imp.load_module(module_name, extfile, filename, ('.py', 'U', 1))
    return None


def run_prolog(pipe, sub_query):
    from my_pyswip import Prolog
    from swip import parse_result
    prolog = Prolog()
    while True:
        request = pipe.recv()
        if request == 0:
            break
        request, args = request
        if request == 'load':
            result = load_external_file(*args, prolog)
        else:
            func = getattr(prolog, request)
            result = func(*args)
        if result is not None:
            result = list(result)
            lr = len(result)
            result = parse_result(result)
            result = result, lr
        pipe.send(result)


class ThreadedProlog(object):

    def __init__(self, sub_query=False):
        pc, cc = Pipe()
        self.pc = pc
        self.p = Process(target=run_prolog, args=(cc, sub_query))
        print(self.p)
        self.p.start()

    def assertz(self, *args):
        self.pc.send(('assertz', args))
        return self.pc.recv()

    def query(self, *args):
        self.pc.send(('query', args))
        return self.pc.recv()

    def consult(self, *args):
        self.pc.send(('consult', args))
        return self.pc.recv()

    def load(self, *args):
        self.pc.send(('load', args))
        return self.pc.recv()

    def __del__(self):
        self.pc.send(0)
        self.p.join()


class DirtyProlog(Prolog):
    def consult(self, filename, *args, **kwargs):
        return super().consult(filename, **kwargs)

    def query(self, filename, *args, **kwargs):
        final = False
        if 'final' in kwargs:
            final = kwargs['final']
        catcherrors = True
        if 'catcherrors' in kwargs:
            catcherrors = kwargs['catcherrors']
        result = super().query(filename, catcherrors=catcherrors)
        if final and result is not None:
            result = list(result)
            lr = len(result)
            result = parse_result(result)
            result = result, lr
        return result

    def load(self, *args, **kwargs):
        return load_external_file(*args, self, **kwargs)
