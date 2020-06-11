from abc import ABC, abstractmethod


class Prolog(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def consult(self, filename: str):
        pass

    @abstractmethod
    def use_module(self, module: str, **kwargs):
        pass

    @abstractmethod
    def asserta(self, clause):
        pass

    @abstractmethod
    def assertz(self, clause):
        pass

    @abstractmethod
    def retract(selfself, clause):
        pass

    @abstractmethod
    def has_solution(self, query):
        pass

    @abstractmethod
    def query(self, *query, **kwargs):
        pass




