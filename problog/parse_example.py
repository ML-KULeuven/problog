from parser import PrologParser, Factory

class List(object) :
    
    def __init__(self, values, tail=None) :
        self.values = values
        self.tail = tail
        
    def __str__(self) :
        if self.tail != None :
            return '[%s|%s]' % (', '.join(map(str,self.values)), self.tail )
        else :
            return '[%s]' % (', '.join(map(str,self.values)))

class Constant(object) :
    
    def __init__(self, value) :
        self.value = value
        
    def __str__(self) :
        return str(self.value)

class Variable(object) :
    
    def __init__(self, name) :
        self.name = name
        
    def __str__(self) :
        return self.name

class Term(object) :
    
    def __init__(self, func, *args) :
        self.func = func
        self.args = args
        self.probability = None
        
    def __str__(self) :
        if self.args :
            s = '%s(%s)' % (self.func, ', '.join(map(str, self.args)))
        else :
            s = '%s' % (self.func)
            
        if self.probability != None :
            s = '%s::%s' % ( self.probability, s)
        return s
            
class Clause(object) :
    
    def __init__(self, head, body, functor=':-') :
        self.head = head
        self.body = body
        self.functor = functor
        
    def __str__(self) :
        return '%s %s %s' % (self.head, self.functor, self.body)
            
class Program(object) :
    
    def __init__(self, clauses) :
        self.clauses = clauses
        
    def __str__(self) :
        return '\n'.join(map(str, self.clauses))
    
    
class SimpleFactory(Factory) :
    """Factory object for creating suitable object from the parse tree."""
        
    def build_program(self, clauses) :
        return Program(clauses)
    
    def build_function(self, functor, arguments) :
        return Term(functor, *arguments)
        
    def build_variable(self, name) :
        return Variable(name)
        
    def build_constant(self, value) :
        return Constant(value)
            
    def build_clause(self, operand1, operand2, functor, **args) :
        return Clause(operand1, operand2, functor)
                        
    def build_probabilistic(self, operand1, operand2, **args) :
        operand2.probability = operand1
        return operand2
                
    def build_list(self, values, tail=None, **extra) :
        return List(values, tail)
            
        
if __name__ == '__main__' :
    from parser import PrologParser
        
    import sys
    if sys.argv[1] == '--string' :
        result = PrologParser(SimpleFactory()).parseString(sys.argv[2])
    else :
        result = PrologParser(SimpleFactory()).parseFile(sys.argv[1])
    print(result)
