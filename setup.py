#! /usr/bin/env python

from __future__ import print_function
import sys
import os


version_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'problog/version.py')

version = {}
with open(version_file) as fp:
    exec(fp.read(), version)
version = version['version']

if __name__ == '__main__' and len(sys.argv) == 1:
    from problog import setup as problog_setup
    problog_setup.install()
elif __name__ == '__main__':

    from setuptools import setup, find_packages
    from setuptools.command.install import install


    class ProbLogInstall(install):
        def run(self):
            install.run(self)
            before_dir = os.getcwd()
            sys.path.insert(0, self.install_lib)
            from problog import setup as problog_setup
            try:
                problog_setup.install()
            except Exception as err:
                print ('Optional ProbLog installation failed: %s' % err, file=sys.stderr)
            os.chdir(before_dir)

    package_data = {
        'problog': [
            'bin/darwin/dsharp', 
            'bin/darwin/maxsatz', 
            'bin/linux/dsharp', 
            'bin/linux/maxsatz',
            'bin/source/maxsatz/maxsatz2009.c',
            'bin/windows/dsharp.exe',
            'bin/windows/maxsatz.exe',
            'bin/windows/libgcc_s_dw2-1.dll',
            'bin/windows/libstdc++-6.dll',
            'lib/sdd/*.h',
            'lib/sdd/*.c',
            'lib/sdd/linux/libsdd.so',
            'lib/sdd/darwin/libsdd.a',
            'lib/sdd/*.py',
            'web/*.py',
            'web/editor_local.html'
            'web/editor_adv.html',
            'web/js/problog_editor.js',
            'library/lists.pl',
            'library/apply.pl',
            'library/control.pl'
        ]
    }


    setup(
        name='problog',
        version=version,
        description='ProbLog2: Probabilistic Logic Programming toolbox',
        url='https://dtai.cs.kuleuven.be/problog',
        author='ProbLog team',
        author_email='anton.dries@cs.kuleuven.be',
        license='Apache Software License',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Prolog',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        keywords='prolog probabilistic logic',
        packages=find_packages(),
        entry_points={
            'console_scripts': ['problog=problog.tasks:main']       
        },
        package_data=package_data,
        cmdclass={
            'install': ProbLogInstall
        }
    )


def increment_release(v):
    v = v.split('.')
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1)]
    else:
        v = v[:4]
    return '.'.join(v)


def increment_dev(v):
    v = v.split('.')
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1), 'dev1']
    else:
        v = v[:4] + ['dev' + str(int(v[4][3:]) + 1)]
    return '.'.join(v)


def increment_version_dev():
    v = increment_dev(version)
    os.path.dirname(__file__)
    with open(version_file, 'w') as f:
        f.write("version = '%s'\n" % v)


def increment_version_release():
    v = increment_release(version)
    with open(version_file, 'w') as f:
        f.write("version = '%s'\n" % v)