from problog.logic import Constant, Term
from problog.pypl import py2pl


distributions = {
    'real' : 'real',

    'delta' : 'delta,',
    'normal' : 'normal',
    'normalMV' : 'normalMV',
    'normalInd' : 'normalInd',
    'beta' : 'beta',
    'poisson' : 'poisson',
    'uniform' : 'uniform',
    'catuni' : 'catuni'
}

pdfs = {
    'real' : 'real',

    'delta' : 'delta,',
    'normal' : 'normal',
    'normalMV' : 'normalMV',
    'normalInd' : 'normalInd',
    'beta' : 'beta',
    'poisson' : 'poisson',
    'uniform' : 'uniform',
    'catuni' : 'catuni'
}

cdfs = {
    'sigmoid' : 'sigmoid'
}

infix_functors = ["/"]
comparison_functors = ["<", ">", "<=", ">="]

class LogicVectorConstant(Term):
    def __init__(self, components):
        assert isinstance(components, list)
        Term.__init__(self, "logicvector", *components)
        self.components = components

    def __add__(self,other):
        operation = "__add__"
        return self._perform_operation(operation, other)
    def __add__(self,other):
        operation = "__radd__"
        return self._perform_operation(operation, other)
    def __sub__(self,other):
        operation = "__sub__"
    def __rsub__(self,other):
        operation = "__rsub__"
    def __mul__(self,other):
        operation = "__mul__"
        return self._perform_operation(operation, other)
    def __rmul__(self,other):
        operation = "__rmul__"
        oepration = self.__mul__
        return self._perform_operation(operation, other)
    def __truediv__(self,other):
        operation = "__truediv__"
        oepration = self.__mul__
        return self._perform_operation(operation, other)
    def __rtruediv__(self,other):
        operation = "__rtruediv__"
        oepration = self.__mul__
        return self._perform_operation(operation, other)
    def __pow__(self,other):
        operation = "__rtruediv__"
        oepration = self.__pow__
        return self._perform_operation(operation, other)
    def __rpow__(self,other):
        operation = "__rtruediv__"
        oepration = self.__rpow__
        return self._perform_operation(operation, other)

    def _perform_operation(self, operation, other):
        components = []
        if isinstance(other, (int, float, SymbolicConstant)):
            for c in self.components:
                op = getattr(c,operation)
                components.append(op(other))
        else:
            assert len(self.components)==len(other.components)
            for c1,c2 in zip(self.components,other.components):
                op = getattr(c1,operation)
                components.append(op(c2))
        return LogicVectorConstant(components)


    def __str__(self):
        return "[" + ",".join(map(str,self.components)) + "]"
    def __repr__(self):
        return str(self)


class SymbolicConstant(Term):
    def __init__(self, value, args=(), cvariables=set()):
        Term.__init__(self, value, *args)
        self.__cvariables = cvariables

    @property
    def cvariables(self):
        return self.__cvariables

    def __add__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "add"
        args = (self,other)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)
    def __radd__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "add"
        args = (other,self)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)

    def __sub__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "sub"
        args = (self,other)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)
    def __rsub__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "sub"
        args = (other,self)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)

    def __mul__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "mul"
        args = (self,other)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)
    def __rmul__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "mul"
        args = (other,self)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)

    def __truediv__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "div"
        args = (self,other)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)
    def __rtruediv__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "div"
        args = (other,self)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)

    def __pow__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "pow"
        args = (self,other)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)
    def __rpow__(self,other):
        if isinstance(other, (int, float)):
            other = SymbolicConstant(other)
        functor = "pow"
        args = (other,self)
        cvariables = set(self.cvariables) | set(other.cvariables)
        return SymbolicConstant(functor, args=args, cvariables=cvariables)



    def __str__(self):
        args = list(map(str,self.args))
        if not args:
            return str(self.functor)
        elif self.functor in infix_functors or self.functor in comparison_functors:
            return "{arg0}{functor}{arg1}".format(functor=self.functor,arg0=args[0], arg1=args[1])
        elif self.functor=="list":
            return "["+",".join(map(str, self.args))+"]"
        else:
            args = ",".join(args)
            return "{functor}({args})".format(functor=self.functor,args=args)
    def __repr__(self):
        return str(self)




class RandomVariableComponentConstant(SymbolicConstant):
    def __init__(self, value, cvariables):
        SymbolicConstant.__init__(self, value, args=(), cvariables=cvariables)

    @property
    def density_name(self):
        return self.functor[:-1]

    @property
    def dimension(self):
        return self.functor[-1]

    def __str__(self):
        return "({},{},{})".format(self.functor[0], self.functor[1], self.functor[2])

    def __repr__(self):
        return str(self)




class RandomVariableConstant(LogicVectorConstant):
    def __init__(self, functor, args, name, dimensions):
        self.name = name
        self.distribution_functor = functor
        self.distribution_args = args

        components = self.make_components(dimensions)
        LogicVectorConstant.__init__(self, components)

    def make_components(self, dimensions):
        components = []
        for d in range(0,dimensions):
            dim_name = self.name+(d,)
            components.append(RandomVariableComponentConstant(dim_name, (dim_name,)))
        return components



class Distribution(Term):
    def __init__(self, distribution, *arguments):
        Term.__init__(self, distribution, *arguments)
        self.dimensions = self.infer_dimensions()

    def infer_dimensions(self):
        return 1



    def __str__(self):
        return str(self.functor)
    def __repr__(self):
        return str(self)
