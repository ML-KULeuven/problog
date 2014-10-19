from __future__ import print_function

# Input python 2 and 3 compatible input
try:
    input = raw_input
except NameError:
    pass


from .unification import TermDB
        
class Engine(object) :
    
    def __init__(self, debugger = None) :
        self.__debugger = debugger
        
        self.__builtins = {}
        
    def getBuiltIn(self, sig) :
        return self.__builtins.get(sig)
        
    def addBuiltIn(self, sig, func) :
        self.__builtins[sig] = func
        
    def _enter_call(self, *args) :
        if self.__debugger != None :
            self.__debugger.enter(*args)
        
    def _exit_call(self, *args) :
        if self.__debugger != None :
            self.__debugger.exit(*args)
                
    def _call_fact(self, db, node, local_vars, tdb, anc, call_key, level) :
        with tdb :
            atoms = list(map( tdb.add, node[1]))
            
            # Match call arguments with fact arguments
            for a,b in zip( local_vars, atoms ) :
                tdb.unify(a,b)
            return ( list(map(tdb.reduce, local_vars)), )
        return []
                    
    def _call_and(self, db, node, local_vars, tdb, anc, call_key, level) :
        with tdb :
            result1 = self._call(db, node[1][0], local_vars, tdb, anc=anc+[call_key], level=level)
            
            result = []
            for res1 in result1 :
                with tdb :
                    for a,b in zip(res1, local_vars) :
                        tdb.unify(a,b)
                    result += self._call(db, node[1][1], local_vars, tdb, anc=anc+[call_key], level=level)
            return result
    
    def _call_or(self, db, node, local_vars, tdb, anc, call_key, level) :
        result = []
        for n in node[1] :
            result += self._call(db, n, local_vars, tdb, anc=anc, level=level) 
        return result
            
    def _call_call(self, db, node, local_vars, tdb, anc, call_key, level) :
        with tdb :
            call_args = [ tdb.add(arg) for arg in node[2] ]
            builtin = self.getBuiltIn( node[3] )
            if builtin != None :
                try :
                    self._enter_call(level, node[3], node[2])
                except UserFail :
                    self._exit_call(level, node[3], node[2], 'USER')
                    return ()
                
                sub = builtin( engine=self, clausedb=db, args=call_args, tdb=tdb, anc=anc+[call_key], level=level)
                
                self._exit_call(level, node[3], node[2], sub)
            else :
                sub = self._call(db, node[1], call_args, tdb, anc=anc+[call_key], level=level)
                
            # result at 
            result = []
            for res in sub :
                with tdb :
                    for a,b in zip(res,call_args) :
                        tdb.unify(a,b)
                    result.append( [ tdb.getTerm(arg) for arg in local_vars ] )
            return result
            
    def _call_not(self, db, node, local_vars, tdb, anc, call_key, level) :
        # TODO Change this for probabilistic
        
        subnode = node[1]
        subresult = self._call(db, subnode, local_vars, tdb, anc, level)
        
        if subresult :
            return []   # no solution
        else :
            return [ [ tdb[v] for v in local_vars ] ]
            
    def _call_def(self, db, node, args, tdb, anc, call_key, level) :
        new_tdb = TermDB()
        
        # Reserve spaces for clause variables
        local_vars = range(0,node[3])
        for i in local_vars :
            new_tdb.newVar()
        
        # Add call arguments to local tdb
        varrename = {}
        call_args = [ tdb.copyTo(arg, new_tdb, varrename) for arg in args ]
                
        with new_tdb :
            # Unify call arguments with head arguments
            for call_arg, def_arg in zip(call_args, node[2]) :
                new_tdb.unify(call_arg, def_arg)
            
            # Call body
            sub = self._call(db, node[1], local_vars, new_tdb, anc, level)
            
            # sub contains values of local variables in the new context 
            #  => we need to translate them back to the 'args' from the old context
            result = []
            for res in sub :
                with tdb :
                    # Unify local_vars with res
                    with new_tdb :
                        for a,b in zip(res, local_vars) :
                            new_tdb.unify(a,b)
                    
                        varrename = {}
                        res_args = [ new_tdb.copyTo(arg, tdb, varrename) for arg in call_args ]
                    
                        # Return values of args
                        result.append( list(map( tdb.getTerm, res_args )))
            return result
            
        return []
    
    def _call_clause(self, db, node, args, tdb, anc, call_key, level) :
        return self._call_or(db, node, args, tdb, anc, call_key, level)
    
    def _call( self, db, node_id, input_vars, tdb, anc=[], level=0) :
        node = db.getNode(node_id)
        
        call_key_1 = []
        call_terms = []
        for arg in input_vars :
            call_terms.append(tdb[arg])
            if not tdb.isGround(arg) :
                call_key_1.append(None)
            else :
                call_key_1.append( str(tdb.getTerm(arg)) )
        call_key = (node_id, tuple(call_key_1) )
        
        try :
            self._enter_call(level, node_id, call_terms)
        except UserFail :
            self._exit_call(level, node_id, call_terms, 'USER')
            return ()
            
        
        if call_key in anc :
            self._exit_call(level, node_id, call_terms, 'CYCLE')
            return ()
        
        rV = None
        if node == None :
            # Undefined
            raise UnknownClause()
        elif node[0] == 'fact' :
            rV = self._call_fact(db, node, input_vars, tdb, anc, call_key,level+1 )
        elif node[0] == 'call' :
            rV = self._call_call(db, node, input_vars, tdb, anc, call_key,level+1)
        elif node[0] == 'def' :
            rV = self._call_def(db, node, input_vars, tdb, anc, call_key,level+1)
        elif node[0] == 'not' :
            rV = self._call_not(db, node, input_vars, tdb, anc, call_key,level+1)
        elif node[0] == 'and' :
            rV = self._call_and(db, node, input_vars, tdb, anc, call_key,level+1)
        elif node[0] == 'or' or node[0] == 'clause':
            rV = self._call_or(db, node, input_vars, tdb, anc, call_key,level+1)
        # elif node[0] == 'builtin' :
        #     rV = self._call_builtin(db, node, input_vars, tdb, anc, call_key,level+1)
        else :
            raise NotImplementedError("Unknown node type: '%s'" % node[0] )
        
        self._exit_call(level, node_id, call_terms, rV)

        return rV

    def query(self, db, term, level=0) :
        tdb = TermDB()
        args = [ tdb.add(x) for x in term.args ]
        clause_node = db.find(term)
        call_node = ('call', clause_node, args, term.signature )
        return self._call_call(db, call_node, args, tdb, [], None, level)
        

class UnknownClause(Exception) : pass

class UserAbort(Exception) : pass

class UserFail(Exception) : pass
        
class Debugger(object) :
        
    def __init__(self, debug=True, trace=False) :
        self.__debug = debug
        self.__trace = trace
        self.__trace_level = None
        
    def enter(self, level, node_id, call_args) :
        if self.__trace :
            print ('  ' * level, '>', node_id, call_args, end='')
            self._trace(level)  
        elif self.__debug :
            print ('  ' * level, '>', node_id, call_args)
        
    def exit(self, level, node_id, call_args, result) :
        
        if not self.__trace and level == self.__trace_level :
            self.__trace = True
            self.__trace_level = None
            
        if self.__trace :
            if result == 'USER' :
                print ('  ' * level, '<', node_id, call_args, result)
            else :
                print ('  ' * level, '<', node_id, call_args, result, end='')
                self._trace(level, False)
        elif self.__debug :
            print ('  ' * level, '<', node_id, call_args, result)
    
    def _trace(self, level, call=True) :
        try : 
            cmd = input('? ')
            if cmd == '' or cmd == 'c' :
                pass    # Do nothing special
            elif cmd.lower() == 's' :
                if call :
                    self.__trace = False
                    if cmd == 's' : self.__debug = False                
                    self.__trace_level = level
            elif cmd.lower() == 'u' :
                self.__trace = False
                if cmd == 'u' : self.__debug = False
                self.__trace_level = level - 1
            elif cmd.lower() == 'l' :
                self.__trace = False
                if cmd == 'l' : self.__debug = False
                self.__trace_level = None
            elif cmd.lower() == 'a' :
                raise UserAbort()
            elif cmd.lower() == 'f' :
                if call :
                    raise UserFail()
            else : # help
                prefix = '  ' * (level) + '    '
                print (prefix, 'Available commands:')
                print (prefix, '\tc\tcreep' )
                print (prefix, '\ts\tskip     \tS\tskip (with debug)' )
                print (prefix, '\tu\tgo up    \tU\tgo up (with debug)' )
                print (prefix, '\tl\tleap     \tL\tleap (with debug)' )
                print (prefix, '\ta\tabort' )
                print (prefix, '\tf\tfail')
                print (prefix, end='')
                self._trace(level,call)
        except EOFError :
            raise UserAbort()
