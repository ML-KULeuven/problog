import org.python.core.PyException;
import org.python.core.PyInteger;
import org.python.core.PyString;
import org.python.core.PyObject;
import org.python.core.PyDictionary;
import org.python.util.PythonInterpreter;

public class ProbLogMain {
    
    public static void main(String[] args) throws PyException {
        
        PythonInterpreter interp = new PythonInterpreter();
        
        interp.exec("import problog");
        interp.set("filename", new PyString(args[0]));
        interp.exec("pl = problog.program.PrologFile(filename)");
        interp.exec("nnf = problog.nnf_formula.NNF.createFrom(pl)");
        interp.exec("res = nnf.evaluate()");
        PyDictionary result = (PyDictionary)interp.get("res");
        System.out.println(result);
    }
    
}