from __future__ import print_function

from .basic import Var

def unify(A,B) :
    tdb = TermDB()
    try :
        tdb.unify(A,B)
        return tdb
    except UnifyError :
        return None

class UnifyError(Exception): pass

class TermDB(object) :
    
    def __init__(self) :
        self.__index = {}
        self.__terms = []
        self.__class = []
        
        self.__undo = []
        
    def _index_set(self, term, index) :
        """Set the index for the given term (undo-able)."""
        assert(not str(term) in self.__index)
        
        self.__index[str(term)] = index
        self._addUndo('i', str(term))
        
    def _index_get(self, term) :
        """Get the index of the given term."""
        return self.__index.get(str(term))
                
    def isGround(self, index) : # used for building call_key
        """Check whether the term at the given position is ground or not."""
        term = self.getTerm(index, recursive=False)
        if term.isVar() :
            return False
        else :
            for arg in term.args :
                if not self.isGround(arg) :
                    return False
        return True
        
    def _terms_add(self, term) :
        """Add a new term and return its index (undo-able)."""
        self.__terms.append(term)
        self.__class.append(len(self.__class))
        self._addUndo('t')
        return len(self.__terms) - 1
        
    def append(self, term) :   # used in top-level call
        """Append a term, without creating a name.""" 
        return self._terms_add(term)
        
    def _terms_get(self, index) :
        """Get class prototype for given index."""
        index = self.__class[index]
        return self.__terms[index]
                                
    def add(self, term) :
        """Add (or lookup) a new term. The argument can also be an index."""
        if type(term) == int :
            return term
        else :
            args = [ self.add(x) for x in term.args ]
            new_term = term.withArgs(*args)
            index = self._index_get(new_term)
            if index == None :
                index = self._terms_add(new_term)
                self._index_set(new_term,index)
            return index
    
    def _newVarName(self) :
        return '$%s_%s' % (id(self), len(self.__terms))
            
    def newVar(self) :  # TODO check for name clash?
        """Create a new variable and return its index."""
        return self._terms_add( Var( self._newVarName() ) )
    
    def find(self, term) :
        """Find the given term and return its index.
            Raises KeyError if the term does not exist.
        """
        if type(term) == int :
            return term
        else :
            args = [ self.find(x) for x in term.args ]
            new_term = term.withArgs(*args)
            index = self._index_get( new_term )
            if index == None :
                raise KeyError(new_term)
            return index
        
    def getTerm(self, index, recursive=True) :
        """Get class prototype term for given index.
            If recursive is True (default), recursively applies this to 
            the term's arguments.
        """
        term = self._terms_get(index)
        if recursive :
            args = [ self.getTerm(x,recursive=True) for x in term.args ]
            term = term.withArgs(*args)
        return term
        
    def copyTo(self, index, db, varrename=None) :
        """Copy the term at the given position to a the given term database.
            Replaces all variables with safe copies.
            
           Arguments:
             varrename - dictionary of old variable to new variable (is updated)
        """
        assert( type(index) == int)
        if varrename == None : varrename = {}
        term = self.getTerm(index, recursive=False)
        if term.isVar() :
            cl = self._getClass(index)
            if cl in varrename :
                return varrename[cl]
            else :
                r = db.newVar()
                varrename[cl] = r
                return r
        else :
            args = [ self.copyTo( arg, db, varrename ) for arg in term.args ]
            new_term = term.withArgs(*args)
            return db.add( new_term )
        
    def reduce(self, term) :
        """Find the class prototype of the given term."""
        try :
            return self.getTerm( self.find(term), recursive=True )
        except KeyError :
            print ('WARNING: term not found', term)
            return term
    
    def __getitem__(self, term) :
        """See reduce"""
        
        return self.reduce(term)
        
    def __str__(self) :
        s = '\n'.join(map(
                            lambda s : '%000d: %s \t\t{%s}' % (s[0], s[1][0], s[1][1]), 
                            enumerate(zip(self.__terms,self.__class))
                        ))
        s += '\n' + str(self.__index)
        return s
        
    def _replaceClass(self, a, b) :
        for i,x in enumerate(self.__class) :
            if x == a :
                self._addUndo('r',i,self.__class[i])
                self.__class[i] = b
    
    def _getClass(self, index) :
        """Get the class of the given index."""
        return self.__class[index]
        
    def _contains(self, iA, iB) :  # TODO why iA == Var?
        """Check if the first argument contains the second.
        The first argument should be a variable.
        """
        # Check if iA contains iB
        assert( self.getTerm(iA, False).isVar() )
        
        pB = self.getTerm(iB, False)
        
        if iA in pB.args :
            return True
        
        for arg in pB.args :
            if self._contains(iA,arg) :
                return True
                
        return False
        
    def _unify_one(self, A, B) :
        """Performs one level of unification and returns list of arguments to
            be unified next.
            Raises UnifyError when unification failed.
        """
        # One level of unification
        
        # Get term index
        if type(A) == int :
            iA = A
        else :
            iA = self.add(A)
        
        if type(B) == int :
            iB = B
        else :
            iB = self.add(B)
        
        # Get classes
        cA = self._getClass(iA)
        cB = self._getClass(iB)
        
        # Already in same class
        if cA == cB : return []
        
        # Get prototype terms
        pA = self.getTerm(cA)
        pB = self.getTerm(cB)
        
        # Merge the two classes
        next_level = []
        
        if pA.isVar() and pB.isVar() :
            self._replaceClass(max(cA,cB),min(cA,cB))
        elif pA.isVar() and not pB.isVar() :
            if self._contains(cA,cB) :   # Detect cyclic definition
                raise UnifyError()
            else :
                self._replaceClass(cA,cB)
        elif pB.isVar() and not pA.isVar() :
            if self._contains(cB,cA) :   # Detect cyclic definition
                raise UnifyError()
            else :
                self._replaceClass(cB,cA)
        else :
            # Both terms are non-var
            if pA.signature == pB.signature :
                self._replaceClass(max(cA,cB),min(cA,cB))
                next_level = list(zip( pA.args, pB.args ))
            else :
                raise UnifyError()   # Incompatible functors
        return next_level
        
    def __enter__(self) :
        self.__undo.append([])
        
    def __exit__(self, err_type, err_value, tb) :
        self._undo()
        if err_type == UnifyError :
            return True
    
    def _addUndo(self, *action) :
        if self.__undo :
            self.__undo[-1].append(action)
    
    def _undo(self) :
        for x in reversed(self.__undo[-1]) :
            if x[0] == 'r' :
                self.__class[x[1]] = x[2]
            elif x[0] == 't' :
                self.__terms.pop(-1)
                self.__class.pop(-1)
            else :
                del self.__index[x[1]]
        self.__undo.pop(-1)
        
    def unify(self, A, B) :
        """Unify terms A and B.
            Raises UnifyError on failure.
        """
        
        queue = self._unify_one(A,B)
        while queue :
            next_one = queue.pop(0)
            queue += self._unify_one( *next_one )
            
        return True
