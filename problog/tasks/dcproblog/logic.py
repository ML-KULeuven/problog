from problog.logic import Term


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




class ValueDimConstant(SymbolicConstant):
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




class ValueExpr(object):
    def __init__(self, functor, args, name, dimensions):
        self.functor = functor
        self.args = args
        self.name = name
        self.dimensions = dimensions
        self.dimension_values = self.make_dim_values(self.dimensions)

    def make_dim_values(self, dimensions):
        dimension_values = []
        for d in range(0,dimensions):
            dim_name = self.name+(d,)
            dimension_values.append(ValueDimConstant(dim_name, (dim_name,)))
            #probably also bug here
        return dimension_values



    def __str__(self):
        return "({},{})".format(self.name[0], self.name[1])

    def __repr__(self):
        return str(self)


class DensityConstant(Term):
    def __init__(self, density):
        Term.__init__(self, density, args=())
    def __str__(self):
        return str(self.functor)
    def __repr__(self):
        return str(self)
