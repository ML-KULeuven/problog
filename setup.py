#! /usr/bin/env python

import sys
import os


version_file = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "problog/version.py"
)

version = {}
with open(version_file) as fp:
    exec(fp.read(), version)
version = version["version"]

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

if __name__ == "__main__" and len(sys.argv) == 1:
    from problog import setup as problog_setup

    problog_setup.install()
elif __name__ == "__main__":

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
                print("Optional ProbLog installation failed: %s" % err, file=sys.stderr)
            os.chdir(before_dir)

    package_data = {
        "problog": [
            "bin/darwin/cnf2dDNNF_wine",
            "bin/darwin/dsharp",
            "bin/darwin/maxsatz",
            "bin/linux/dsharp",
            "bin/linux/maxsatz",
            "bin/source/maxsatz/maxsatz2009.c",
            "bin/windows/dsharp.exe",
            "bin/windows/maxsatz.exe",
            "bin/windows/libgcc_s_dw2-1.dll",
            "bin/windows/libstdc++-6.dll",
            "web/*.py",
            "web/editor_local.html" "web/editor_adv.html",
            "web/js/problog_editor.js",
            "library/*.pl",
            "library/*.py",
            "library/nlp4plp.d/*",
        ]
    }

    setup(
        name="problog",
        version=version,
        description="ProbLog2: Probabilistic Logic Programming toolbox",
        long_description=long_description,
        long_description_content_type='text/markdown',
        url="https://dtai.cs.kuleuven.be/problog",
        author="ProbLog team",
        author_email="anton.dries@cs.kuleuven.be",
        license="Apache Software License",
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Prolog",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords="prolog probabilistic logic",
        packages=find_packages(),
        install_requires = ["setuptools"],
        extras_require={"sdd": ["pysdd>=0.2.6"]},
        entry_points={"console_scripts": ["problog=problog.tasks:main"]},
        package_data=package_data,
        cmdclass={"install": ProbLogInstall},
    )


def increment_release(v):
    v = v.split(".")
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1)]
    else:
        v = v[:4]
    return ".".join(v)


def increment_dev(v):
    v = v.split(".")
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1), "dev1"]
    else:
        v = v[:4] + ["dev" + str(int(v[4][3:]) + 1)]
    return ".".join(v)


def increment_version_dev():
    v = increment_dev(version)
    os.path.dirname(__file__)
    with open(version_file, "w") as f:
        f.write("version = '%s'\n" % v)


def increment_version_release():
    v = increment_release(version)
    with open(version_file, "w") as f:
        f.write("version = '%s'\n" % v)
