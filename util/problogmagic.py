"""
ProbLog IPython magic extensions

Magic methods:
    %problog <model>
    %%problog <model ...
    ... >
    %problogstr "<dot graph>"
    %problogmodel model
    %problogmodels [model1, model2, ...]

Usage:

    %load_ext problogmagic

Based on:
https://gist.github.com/cjdrake/7982333
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../examples'))

from problog.core import ProbLog
from problog.program import PrologString
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
import example_sampling_alt as plsample

if SDD.is_available():
    knowledge = SDD
else:
    knowledge = NNF

from IPython.core.display import display_html
from IPython.core.magic import (
    Magics, magics_class,
    line_magic, line_cell_magic
)
from IPython.core.magic_arguments import (
    argument, magic_arguments,
    parse_argstring)
from IPython.utils.warn import info, error


def runproblog(s,output='html'):
    """Execute problog and return an html snippet, or None."""
    model = PrologString(s)
    try:
      result = ProbLog.convert(model, knowledge).evaluate()
    except Exception as e:
        return '<pre>{}</pre>'.format(e)
    if result is None:
        error('Error when running ProbLog')
        return None
    else:
        return formatoutput(result, output=output)

def runproblogsampling(s,n=5,output='html'):
    from example_sampling_alt import print_result

    model = PrologString(s)
    samples, db = plsample.sample_object(model, N=n)
    result = ''
    for sample in samples:
        result += sample.toString(db,False,True)+'<br>'
    if output == 'html':
        return '<pre>{}</pre>'.format(result)
    return result


def formatoutput(result, output='html'):
    if output == 'html':
        html = '<table style="width:100%;"><tr><th style="width:66%;">Atom<th>Probability'
        atomprobs = [(str(atom),prob) for atom,prob in result.items()]
        atomprobs.sort(key=lambda x:x[0])
        for atom,prob in atomprobs:
            p = prob*100
            html += '<tr><td>{a}'.format(a=atom)
            if p < 10:
                color = 'black'
            else:
                color = 'white'
            html += '<td><div class="progress-bar" role="progressbar" aria-valuenow="{perc}" aria-valuemin="0" aria-valuemax="100" style="text-align:left;width:{perc}%;padding:3px;color:{c};">{prob:6.4f}</div>'.format(perc=prob*100,prob=prob,c=color)
        html += '</table>'
        return html
    else:
        txt = ''
        for atom,prob in result.items():
            txt += '{:<40}: {}'.format(atom,prob)
        return txt


@magics_class
class ProbLogMagics(Magics):

    @line_cell_magic
    def problog(self, line, cell=None):
        """problog line/cell magic"""
        if cell is None:
            s = line
        else:
            s = line + '\n' + cell
        data = runproblog(s, output='html')
        if data:
            display_html(data, raw=True)

    @line_magic
    def problogstr(self, line):
        """problog string magic"""
        s = self.shell.ev(line)
        data = runproblog(s, output='html')
        if data:
            display_html(data, raw=True)

    @line_magic
    def problogobj(self, line):
        """problog object magic"""
        obj = self.shell.ev(line)
        if not isinstance(obj, LogicProgram):
            error("expected object to be of type LogicProgram")
        else:
            data = runproblog(s, output='html')
            if data:
                display_html(data, raw=True)

    @line_magic
    def problogobjs(self, line):
        """problog objects magic"""
        objs = self.shell.ev(line)
        for i, obj in enumerate(objs):
            if not isinstance(obj, LogicProgram):
                error("expected object {} to be of type LogicProgram".format(i))
            else:
                data = runproblog(s, output='html')
                if data:
                    info('object {}:'.format(i))
                    display_html(data, raw=True)

    @line_cell_magic
    @magic_arguments()
    @argument('-N', help='Number of samples')
    def problogsample(self, line, cell=None):
        """problog line/cell magic"""
        args = parse_argstring(self.problogsample, line)
        n = 5
        if not args.N is None:
            n = int(args.N)
        s = cell
        data = runproblogsampling(s, n=n, output='html')
        if data:
            display_html(data, raw=True)


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(ProbLogMagics)

