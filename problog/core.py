from __future__ import print_function

from collections import defaultdict

import traceback

LABEL_QUERY = "query"
LABEL_EVIDENCE_POS = "evidence+"
LABEL_EVIDENCE_NEG = "evidence-"
LABEL_EVIDENCE_MAYBE = "evidence?"
LABEL_NAMED = "named"


class TransformationUnavailable(Exception):
    pass


class ProbLog(object) :
    """This is a static class"""
    
    def __init__(self) :
        raise Exception("This is a static class!")
        
    transformations = defaultdict(list)    
        
    @classmethod
    def registerTransformation(cls, src, target, action=None) :
        cls.transformations[target].append( (src, action) )
    
    @classmethod
    def findPaths( cls, src, target, stack=() ) :
        # Create a destination object or any of its subclasses
        if isinstance(src, target) :
            yield (target,)
        else :
            # for d in list(cls.transformations) :
            #     if issubclass(d,target) :
                    d = target
                    for s, action in cls.transformations[d] :
                        if not s in stack :                        
                            for path in cls.findPaths( src, s, stack+(s,) ) :
                                yield path + (action,d)
    
    @classmethod
    def convert(cls, src, target, **kwdargs):
        """
        Convert the source object into an object of the target class.
        :param src: source object
        :param target: target class
        :param kwdargs: additional arguments passed to transformation functions
        """
        # Find transformation paths from source to target.
        for path in cls.findPaths(src, target):
            try:
                # A path is a sequence of obj, function, obj/class, function, obj/class, ..., obj_class
                current_obj = src
                path = path[1:]
                # Invariant: path[0] is a function, path[1] is an obj/class
                while path:
                    if path[1] is not None:
                        next_obj = path[0](current_obj, path[1](**kwdargs), **kwdargs)
                    else:
                        next_obj = path[1].createFromDefaultAction(current_obj, **kwdargs)
                    path = path[2:]
                    current_obj = next_obj
                return current_obj
            except TransformationUnavailable:
                # The current transformation strategy didn't work for some reason. Try another one.
                pass
        raise ProbLogError("No conversion strategy found from an object of class '%s' to an object of class '%s'."
                           % (type(src).__name__, target.__name__))


class ProbLogError(Exception):
    """General Problog error."""
    pass

class ParseError(ProbLogError) : pass 


class GroundingError(ProbLogError):
    """Represents an error that occurred during grounding."""

    def __init__(self, base_message, location=None):
        self.base_message = base_message
        self.location = location

    def _location_string(self):
        if self.location is None:
            return ''
        if type(self.location) == tuple:
            return ' at %s:%s' % self.location
        else:
            return ' at character %s' % self.location

    def _message(self):
        return '%s: %s%s.' % (self.__class__.__name__, self.base_message, self._location_string())

    def __str__(self):
        return self._message()


class CompilationError(ProbLogError) : pass

def process_error( err, debug=False ) :
    """Take the given error raise by ProbLog and produce a meaningful error message."""
    
    if debug :
        traceback.print_exc()

    err_type = type(err).__name__
    if err_type == 'ParseException' :
        return 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line )
    elif isinstance(err, ParseError) :
        return 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line )
    elif isinstance(err, GroundingError) :
        return 'Error during grounding: %s' % err
    elif isinstance(err, CompilationError) :
        return 'Error during compilation: %s' % err
    elif isinstance(err, ProbLogError) :
        return 'Error: %s' % err
    else :
        if not debug : traceback.print_exc()
        return 'Unknown error: %s' % (err_type)

class ProbLogObject(object) :
    """Root class for all convertible objects in the ProbLog system."""


    @classmethod
    def createFrom(cls, obj, **kwdargs):
        return ProbLog.convert(obj, cls, **kwdargs)

    @classmethod
    def createFromDefaultAction(cls, src) :
        raise ProbLogError("No default conversion strategy defined.")

class transform(object) :
    """Decorator"""
    
    def __init__(self, cls1, cls2, func=None) :
        self.cls1 = cls1
        self.cls2 = cls2
        if not issubclass(cls2, ProbLogObject) :
            raise TypeError("Conversion only possible for subclasses of ProbLogObject.")        
        if func != None :
            self(func)
        
    def __call__(self, f) :
        # TODO check type contract?
        ProbLog.registerTransformation( self.cls1, self.cls2, f )
        return f
        
def list_transformations() :
    print ('Available transformations:')
    for target in ProbLog.transformations :
        print ('\tcreate %s.%s' % (target.__module__, target.__name__) )
        for src, func in ProbLog.transformations[target] :
            print ('\t\tfrom %s.%s by %s.%s' % (src.__module__, src.__name__, func.__module__, func.__name__) )