import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

dcproblog_tasks = {}
dcproblog_tasks['inference'] = 'dcproblog.tasks.inference'

dcproblog_default_task = 'inference'

from problog.util import load_module


def load_task(name):
    """Load the module for executing the given task.

    :param name: task name
    :type name: str
    :return: loaded module
    :rtype: module
    """
    return load_module(dcproblog_tasks[name])

def run_task(argv):
    """Execute a task in DC-ProbLog.
    If the first argument is a known task name, that task is executed.
    Otherwise the default task is executed.

    :param argv: list of arguments for the task
    :return: result of the task (typically None)
    """
    if len(argv) > 0 and argv[0] in dcproblog_tasks:
        task = argv[0]
        args = argv[1:]
    else:
        task = dcproblog_default_task
        args = argv
    return load_task(task).main(args)


def main(*args):
    argv = sys.argv[1:]
    run_task(argv[1:])
