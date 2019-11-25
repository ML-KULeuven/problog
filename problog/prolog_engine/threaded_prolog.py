from multiprocessing import Process, Pipe
from pyswip.prolog import Prolog


def run_prolog(pipe):
    prolog = Prolog()
    while True:
        request = pipe.recv()
        if request == 0:
            break
        request, args = request
        func = getattr(prolog, request)
        result = func(*args)
        if result is not None:
            result = list(result)
        pipe.send(result)


class ThreadedProlog(object):

    def __init__(self):
        pc, cc = Pipe()
        self.pc = pc
        self.p = Process(target=run_prolog, args=(cc,))
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

    def __del__(self):
        self.pc.send(0)
        self.p.join()
