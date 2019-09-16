The Sentential Decision Diagram Package
sdd version 2.0, January 8, 2018
http://reasoning.cs.ucla.edu/sdd

This directory provides the source code for the SDD library.

All source code provided is in the C programming language (and in
particular, C99).

COMPILATION

The SDD library uses the SCons build tool for compiling the library
(see http://scons.org).  If SCons is installed on the system, the
library can be compiled by running the following on the command line:

  scons

(run this command in the same directory that contains the file
SConstruct).  This command compiles a static library (named libsdd.a)
and a shared library (named libsdd.so on Linux, and named libsdd.dylib
on Mac).  Both will be found under the build directory.  The following
command:

  scons mode=debug

produces libraries with debugging information included, found under
the debug directory.  Adding a -c flag to either of the above commands
will clean the respective build.

The debug build will enable assertions by default.  To disable these
assertions, run the command:

  scons mode=debug --disable-assertions

A more expensive but more exhaustive debugging mode can be enabled by
running the following command:

  scons mode=debug --enable-full-debug

These options are ignored when compiling the library without the
mode=debug option.

AUTHORS

The SDD Package was developed by Arthur Choi and Adnan Darwiche, of
the Automated Reasoning Group at the University of California, Los
Angeles.

  http://reasoning.cs.ucla.edu

Feedback, bug reports, and questions can be sent to the email address

  sdd@cs.ucla.edu

