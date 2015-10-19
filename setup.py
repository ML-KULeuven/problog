#! /usr/bin/env python

from __future__ import print_function
import sys

version = '2.1.0.6'

if len(sys.argv) == 1:
    from problog import setup as problog_setup
    problog_setup.install()
else:    

    from setuptools import setup, find_packages
    from setuptools.command.install import install


    class ProbLogInstall(install):
        def run(self):
            install.run(self)
            from problog import setup as problog_setup
            try:
                problog_setup.install()
            except Exception as err:
                print ('Optional ProbLog installation failed: %s' % err, file=sys.stderr)

    package_data = {
        'problog': [
            'bin/darwin/dsharp', 
            'bin/darwin/maxsatz', 
            'bin/linux/dsharp', 
            'bin/linux/maxsatz',
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
            'web/editor_local.html',
            'web/editor_adv.html',
            'web/js/problog_editor.js'
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
            'License :: OSI Approved :: Apple Public Source License',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
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


