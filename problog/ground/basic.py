from __future__ import print_function

from ..utils import TemporaryDirectory, local_path

from ..logic.program import PrologFile




class Grounder(object) :
    
    def __init__(self, env=None) :
        self.env = env        
        
    def ground(self, lp, update=None) :
        """Ground the given logic program.
            :param lp: logic program
            :type lp: :class:`.LogicProgram`
            :param update: ground program to update (incremental grounding)
            :type update: :class:`.GroundProgram`
            :returns: grounded version of the given logic program
            :rtype: :class:`.GroundProgram`
        """
        raise NotImplementedError("Grounder.ground is an abstract method." )
        
        
# Taken and modified from ProbFOIL
class YapGrounder(Grounder) :
    
    def __init__(self, env=None) :
        Grounder.__init__(self, env)
    
    def ground(self, lp, update=None) :
        # This grounder requires a physical Prolog file.
        lp = PrologFile.createFrom(lp)
        
        # Call the grounder.
        grounder_result, qr = self._call_grounder(lp.filename)
        
        # qr are queries (list)
        
        # Initialize a new ground program if none was given.
        if update == None : update = GroundProgram()
        
        # Integrate result into the ground program
        #res = update.integrate(grounder_result)

        # Return the result
        return update
    
    def _call_grounder(self, in_file) : 
        PROBLOG_GROUNDER = local_path('ground/ground_compact.pl')
        
        # Create a temporary directory that is removed when the block exits.
        with TemporaryDirectory(tmpdir='/tmp/problog/') as tmp :
                
            # 2) Call yap to do the actual grounding
            ground_program = tmp.abspath('problog.ground')
        
            queries = tmp.abspath('problog.queries')
        
            evidence = tmp.abspath('problog.evidence')
                
            import subprocess
        
            try :
                output = subprocess.check_output(['yap', "-L", PROBLOG_GROUNDER , '--', in_file, ground_program, evidence, queries ])
                with open(queries) as f :
                    qr = [ line.strip() for line in f ]
                    
                self._build_grounding(ground_program)
                
                return None, None    
               # return self._read_grounding(ground_program), qr
            except subprocess.CalledProcessError :
                print ('Error during grounding', file=sys.stderr)
                return [], []
    
    def _read_grounding(self, filename) :
        lines = []
        with open(filename,'r') as f :
            for line in f :
                line = line.strip().split('|', 1)
                name = None
            
                if len(line) > 1 : name = line[1].strip()
                line = line[0].split()
                line_id = int(line[0])
                line_type = line[1].lower()
            
                while line_id >= len(lines) :
                    lines.append( (None,[],None) )
                if line_type == 'fact' :
                    line_content = float(line[2])
                else :
                    line_content = lines[line_id][1] + [(line_type, line[2:])]
                    line_type = 'or'
                
                lines[line_id] = ( line_type, line_content, name )
        return lines
        
    def _build_grounding(self, filename) :
        # line = 'line_type', 'line_content', 'name'
        
        
        for line in open(filename) :
            content, name = line.strip().split(' | ')
            content = content.split()
            
            line_index = content[0]
            line_type = content[1]
            line_content = content[2:]
        
            
            
        
        pass
        
        

# Grounder with constraints


