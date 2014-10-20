import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

from problog.logic.program import PrologFile
from problog.ground.basic import YapGrounder

def main(filename) :
    
    # lp = LogicProgram()
    lp = PrologFile(os.path.abspath(filename))
    
    #gr = Grounder()
    gr = YapGrounder()
    
    gp = gr.ground(lp)
    
    print (gp)
    
    # kc = Compiler()
    #
    # cp = kc.compile(gp)
    #
    # ev = Evaluator()
    #
    # rs = Evaluate(cp)
    #
    # print (rs)
    
if __name__ == '__main__' :
    main(*sys.argv[1:])
    

