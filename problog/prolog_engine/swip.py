import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile



def run_string(string, query):
    with NamedTemporaryFile('w') as tfile:
        tfile.write(string)
        tfile.seek(0)
        return run_file(tfile.name, query)

def run_file(path, query):
    path = Path(path)
    out = subprocess.run(['swipl', '-q', '-l', str(path), '-g', query, '-t', 'halt'], capture_output=True, encoding='ascii')
    return out.stdout.strip()