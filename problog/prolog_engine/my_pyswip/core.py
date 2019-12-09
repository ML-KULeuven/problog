# -*- coding: utf-8 -*-


# pyswip -- Python SWI-Prolog bridge
# Copyright (c) 2007-2018 Yüce Tekol
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import os
import sys
import glob
import warnings
import atexit
from subprocess import Popen, PIPE
from ctypes import *
from ctypes.util import find_library
import tempfile
import shutil


# To initialize the SWI-Prolog environment, two things need to be done: the
# first is to find where the SO/DLL is located and the second is to find the
# SWI-Prolog home, to get the saved state.
#
# The goal of the (entangled) process below is to make the library installation
# independent.

def create_temporary_copy(path):
    temp_dir = tempfile.gettempdir()
    temp = tempfile.NamedTemporaryFile(dir=temp_dir)
    temp_path = os.path.join(temp_dir, temp.name)
    shutil.copy2(path, temp_path)
    return temp, temp_path


class SWIPl:
    def __init__(self):
        # Find the path and resource file. SWI_HOME_DIR shall be treated as a constant
        # by users of this module
        self._path, self.SWI_HOME_DIR = SWIPl._findSwipl()
        SWIPl._fixWindowsPath(self._path)

        temp, temp_path = create_temporary_copy(self._path)
        # Load the library
        self._lib = CDLL(temp_path)
        temp.close()

        self.PL_initialise = self._lib.PL_initialise
        self.PL_initialise = check_strings(None, 1)(self.PL_initialise)
        # PL_initialise.argtypes = [c_int, c_c??

        self.PL_open_foreign_frame = self._lib.PL_open_foreign_frame
        self.PL_open_foreign_frame.restype = fid_t

        self.PL_foreign_control = self._lib.PL_foreign_control
        self.PL_foreign_control.argtypes = [control_t]
        self.PL_foreign_control.restype = c_int

        self.PL_foreign_context = self._lib.PL_foreign_context
        self.PL_foreign_context.argtypes = [control_t]
        self.PL_foreign_context.restype = intptr_t

        self.PL_retry = self._lib._PL_retry
        self.PL_retry.argtypes = [intptr_t]
        self.PL_retry.restype = foreign_t

        self.PL_new_term_ref = self._lib.PL_new_term_ref
        self.PL_new_term_ref.restype = term_t

        self.PL_new_term_refs = self._lib.PL_new_term_refs
        self.PL_new_term_refs.argtypes = [c_int]
        self.PL_new_term_refs.restype = term_t

        self.PL_chars_to_term = self._lib.PL_chars_to_term
        self.PL_chars_to_term.argtypes = [c_char_p, term_t]
        self.PL_chars_to_term.restype = c_int

        self.PL_chars_to_term = check_strings(0, None)(self.PL_chars_to_term)

        self.PL_call = self._lib.PL_call
        self.PL_call.argtypes = [term_t, module_t]
        self.PL_call.restype = c_int

        self.PL_call_predicate = self._lib.PL_call_predicate
        self.PL_call_predicate.argtypes = [module_t, c_int, predicate_t, term_t]
        self.PL_call_predicate.restype = c_int

        self.PL_discard_foreign_frame = self._lib.PL_discard_foreign_frame
        self.PL_discard_foreign_frame.argtypes = [fid_t]
        self.PL_discard_foreign_frame.restype = None

        self.PL_put_list_chars = self._lib.PL_put_list_chars
        self.PL_put_list_chars.argtypes = [term_t, c_char_p]
        self.PL_put_list_chars.restype = c_int

        self.PL_put_list_chars = check_strings(1, None)(self.PL_put_list_chars)

        # PL_EXPORT(void)                PL_register_atom(atom_t a);
        self.PL_register_atom = self._lib.PL_register_atom
        self.PL_register_atom.argtypes = [atom_t]
        self.PL_register_atom.restype = None

        # PL_EXPORT(void)                PL_unregister_atom(atom_t a);
        self.PL_unregister_atom = self._lib.PL_unregister_atom
        self.PL_unregister_atom.argtypes = [atom_t]
        self.PL_unregister_atom.restype = None

        # PL_EXPORT(atom_t)      PL_functor_name(functor_t f);
        self.PL_functor_name = self._lib.PL_functor_name
        self.PL_functor_name.argtypes = [functor_t]
        self.PL_functor_name.restype = atom_t

        # PL_EXPORT(int)         PL_functor_arity(functor_t f);
        self.PL_functor_arity = self._lib.PL_functor_arity
        self.PL_functor_arity.argtypes = [functor_t]
        self.PL_functor_arity.restype = c_int

        #                       /* Get C-values from Prolog terms */
        # PL_EXPORT(int)         PL_get_atom(term_t t, atom_t *a);
        self.PL_get_atom = self._lib.PL_get_atom
        self.PL_get_atom.argtypes = [term_t, POINTER(atom_t)]
        self.PL_get_atom.restype = c_int

        # PL_EXPORT(int)         PL_get_bool(term_t t, int *value);
        self.PL_get_bool = self._lib.PL_get_bool
        self.PL_get_bool.argtypes = [term_t, POINTER(c_int)]
        self.PL_get_bool.restype = c_int

        # PL_EXPORT(int)         PL_get_atom_chars(term_t t, char **a);
        self.PL_get_atom_chars = self._lib.PL_get_atom_chars  # FIXME
        self.PL_get_atom_chars.argtypes = [term_t, POINTER(c_char_p)]
        self.PL_get_atom_chars.restype = c_int

        self.PL_get_atom_chars = check_strings(None, 1)(self.PL_get_atom_chars)

        self.PL_get_string_chars = self._lib.PL_get_string
        self.PL_get_string_chars.argtypes = [term_t, POINTER(c_char_p), c_int_p]

        # PL_EXPORT(int)         PL_get_chars(term_t t, char **s, unsigned int flags);
        self.PL_get_chars = self._lib.PL_get_chars  # FIXME:

        self.PL_get_chars = check_strings(None, 1)(self.PL_get_chars)

        # PL_EXPORT(int)         PL_get_list_chars(term_t l, char **s,
        #                                         unsigned int flags);
        # PL_EXPORT(int)         PL_get_atom_nchars(term_t t, size_t *len, char **a);
        # PL_EXPORT(int)         PL_get_list_nchars(term_t l,
        #                                          size_t *len, char **s,
        #                                          unsigned int flags);
        # PL_EXPORT(int)         PL_get_nchars(term_t t,
        #                                     size_t *len, char **s,
        #                                     unsigned int flags);
        # PL_EXPORT(int)         PL_get_integer(term_t t, int *i);
        self.PL_get_integer = self._lib.PL_get_integer
        self.PL_get_integer.argtypes = [term_t, POINTER(c_int)]
        self.PL_get_integer.restype = c_int

        # PL_EXPORT(int)         PL_get_long(term_t t, long *i);
        self.PL_get_long = self._lib.PL_get_long
        self.PL_get_long.argtypes = [term_t, POINTER(c_long)]
        self.PL_get_long.restype = c_int

        # PL_EXPORT(int)         PL_get_pointer(term_t t, void **ptr);
        # PL_EXPORT(int)         PL_get_float(term_t t, double *f);
        self.PL_get_float = self._lib.PL_get_float
        self.PL_get_float.argtypes = [term_t, c_double_p]
        self.PL_get_float.restype = c_int

        # PL_EXPORT(int)         PL_get_functor(term_t t, functor_t *f);
        self.PL_get_functor = self._lib.PL_get_functor
        self.PL_get_functor.argtypes = [term_t, POINTER(functor_t)]
        self.PL_get_functor.restype = c_int

        # PL_EXPORT(int)         PL_get_name_arity(term_t t, atom_t *name, int *arity);
        self.PL_get_name_arity = self._lib.PL_get_name_arity
        self.PL_get_name_arity.argtypes = [term_t, POINTER(atom_t), POINTER(c_int)]
        self.PL_get_name_arity.restype = c_int

        # PL_EXPORT(int)         PL_get_module(term_t t, module_t *module);
        # PL_EXPORT(int)         PL_get_arg(int index, term_t t, term_t a);
        self.PL_get_arg = self._lib.PL_get_arg
        self.PL_get_arg.argtypes = [c_int, term_t, term_t]
        self.PL_get_arg.restype = c_int

        # PL_EXPORT(int)         PL_get_list(term_t l, term_t h, term_t t);
        # PL_EXPORT(int)         PL_get_head(term_t l, term_t h);
        self.PL_get_head = self._lib.PL_get_head
        self.PL_get_head.argtypes = [term_t, term_t]
        self.PL_get_head.restype = c_int

        # PL_EXPORT(int)         PL_get_tail(term_t l, term_t t);
        self.PL_get_tail = self._lib.PL_get_tail
        self.PL_get_tail.argtypes = [term_t, term_t]
        self.PL_get_tail.restype = c_int

        # PL_EXPORT(int)         PL_get_nil(term_t l);
        self.PL_get_nil = self._lib.PL_get_nil
        self.PL_get_nil.argtypes = [term_t]
        self.PL_get_nil.restype = c_int

        # PL_EXPORT(int)         PL_get_term_value(term_t t, term_value_t *v);
        # PL_EXPORT(char *)      PL_quote(int chr, const char *data);

        self.PL_put_atom_chars = self._lib.PL_put_atom_chars
        self.PL_put_atom_chars.argtypes = [term_t, c_char_p]
        self.PL_put_atom_chars.restype = c_int

        self.PL_put_atom_chars = check_strings(1, None)(self.PL_put_atom_chars)

        self.PL_atom_chars = self._lib.PL_atom_chars
        self.PL_atom_chars.argtypes = [atom_t]
        self.PL_atom_chars.restype = c_char_p

        self.PL_predicate = self._lib.PL_predicate
        self.PL_predicate.argtypes = [c_char_p, c_int, c_char_p]
        self.PL_predicate.restype = predicate_t

        self.PL_predicate = check_strings([0, 2], None)(self.PL_predicate)

        self.PL_pred = self._lib.PL_pred
        self.PL_pred.argtypes = [functor_t, module_t]
        self.PL_pred.restype = predicate_t

        self.PL_open_query = self._lib.PL_open_query
        self.PL_open_query.argtypes = [module_t, c_int, predicate_t, term_t]
        self.PL_open_query.restype = qid_t

        self.PL_next_solution = self._lib.PL_next_solution
        self.PL_next_solution.argtypes = [qid_t]
        self.PL_next_solution.restype = c_int

        self.PL_copy_term_ref = self._lib.PL_copy_term_ref
        self.PL_copy_term_ref.argtypes = [term_t]
        self.PL_copy_term_ref.restype = term_t

        self.PL_get_list = self._lib.PL_get_list
        self.PL_get_list.argtypes = [term_t, term_t, term_t]
        self.PL_get_list.restype = c_int

        self.PL_get_chars = self._lib.PL_get_chars  # FIXME

        self.PL_close_query = self._lib.PL_close_query
        self.PL_close_query.argtypes = [qid_t]
        self.PL_close_query.restype = None

        # void PL_cut_query(qid)
        self.PL_cut_query = self._lib.PL_cut_query
        self.PL_cut_query.argtypes = [qid_t]
        self.PL_cut_query.restype = None

        self.PL_halt = self._lib.PL_halt
        self.PL_halt.argtypes = [c_int]
        self.PL_halt.restype = None

        # PL_EXPORT(int)        PL_cleanup(int status);
        self.PL_cleanup = self._lib.PL_cleanup
        self.PL_cleanup.restype = c_int

        self.PL_unify_integer = self._lib.PL_unify_integer
        self.PL_unify = self._lib.PL_unify
        self.PL_unify.argtypes = [term_t, term_t]
        self.PL_unify.restype = c_int

        self.PL_unify_atom = self._lib.PL_unify_atom
        self.PL_unify_atom.argtypes = [term_t, atom_t]
        self.PL_unify_atom.restype = c_int

        self.PL_unify_float = self._lib.PL_unify_float
        self.PL_unify_float.argtypes = [term_t, c_double]
        self.PL_unify_float.restype = c_int

        # PL_EXPORT(int)          PL_unify_arg(int index, term_t t, term_t a) WUNUSED;
        self.PL_unify_arg = self._lib.PL_unify_arg
        self.PL_unify_arg.argtypes = [c_int, term_t, term_t]
        self.PL_unify_arg.restype = c_int

        self.PL_unify_list = self._lib.PL_unify_list
        self.PL_unify_list.argtypes = [term_t, term_t, term_t]
        self.PL_unify_list.restype = c_int

        self.PL_unify_nil = self._lib.PL_unify_nil
        self.PL_unify_nil.argtypes = [term_t]
        self.PL_unify_nil.restype = c_int

        self.PL_unify_functor = self._lib.PL_unify_functor
        self.PL_unify_functor.argtypes = [term_t, functor_t]
        self.PL_unify_functor.restype = c_int

        # Verify types

        self.PL_term_type = self._lib.PL_term_type
        self.PL_term_type.argtypes = [term_t]
        self.PL_term_type.restype = c_int

        self.PL_is_variable = self._lib.PL_is_variable
        self.PL_is_variable.argtypes = [term_t]
        self.PL_is_variable.restype = c_int

        self.PL_is_ground = self._lib.PL_is_ground
        self.PL_is_ground.argtypes = [term_t]
        self.PL_is_ground.restype = c_int

        self.PL_is_atom = self._lib.PL_is_atom
        self.PL_is_atom.argtypes = [term_t]
        self.PL_is_atom.restype = c_int

        self.PL_is_integer = self._lib.PL_is_integer
        self.PL_is_integer.argtypes = [term_t]
        self.PL_is_integer.restype = c_int

        self.PL_is_string = self._lib.PL_is_string
        self.PL_is_string.argtypes = [term_t]
        self.PL_is_string.restype = c_int

        self.PL_is_float = self._lib.PL_is_float
        self.PL_is_float.argtypes = [term_t]
        self.PL_is_float.restype = c_int

        # PL_is_rational = _lib.PL_is_rational
        # PL_is_rational.argtypes = [term_t]
        # PL_is_rational.restype = c_int

        self.PL_is_compound = self._lib.PL_is_compound
        self.PL_is_compound.argtypes = [term_t]
        self.PL_is_compound.restype = c_int

        self.PL_is_functor = self._lib.PL_is_functor
        self.PL_is_functor.argtypes = [term_t, functor_t]
        self.PL_is_functor.restype = c_int

        self.PL_is_list = self._lib.PL_is_list
        self.PL_is_list.argtypes = [term_t]
        self.PL_is_list.restype = c_int

        self.PL_is_atomic = self._lib.PL_is_atomic
        self.PL_is_atomic.argtypes = [term_t]
        self.PL_is_atomic.restype = c_int

        self.PL_is_number = self._lib.PL_is_number
        self.PL_is_number.argtypes = [term_t]
        self.PL_is_number.restype = c_int

        #                       /* Assign to term-references */
        # PL_EXPORT(void)                PL_put_variable(term_t t);
        self.PL_put_variable = self._lib.PL_put_variable
        self.PL_put_variable.argtypes = [term_t]
        self.PL_put_variable.restype = None

        # PL_EXPORT(void)                PL_put_atom(term_t t, atom_t a);
        # PL_EXPORT(void)                PL_put_atom_chars(term_t t, const char *chars);
        # PL_EXPORT(void)                PL_put_string_chars(term_t t, const char *chars);
        # PL_EXPORT(void)                PL_put_list_chars(term_t t, const char *chars);
        # PL_EXPORT(void)                PL_put_list_codes(term_t t, const char *chars);
        # PL_EXPORT(void)                PL_put_atom_nchars(term_t t, size_t l, const char *chars);
        # PL_EXPORT(void)                PL_put_string_nchars(term_t t, size_t len, const char *chars);
        # PL_EXPORT(void)                PL_put_list_nchars(term_t t, size_t l, const char *chars);
        # PL_EXPORT(void)                PL_put_list_ncodes(term_t t, size_t l, const char *chars);
        self.PL_put_integer = self._lib.PL_put_integer
        self.PL_put_integer.argtypes = [term_t, c_long]
        self.PL_put_integer.restype = None

        # PL_EXPORT(void)                PL_put_pointer(term_t t, void *ptr);
        self.PL_put_float = self._lib.PL_put_float
        self.PL_put_float.argtypes = [term_t, c_double]
        self.PL_put_float.restype = None

        self.PL_put_functor = self._lib.PL_put_functor
        self.PL_put_functor.argtypes = [term_t, functor_t]
        self.PL_put_functor.restype = None

        # PL_EXPORT(void)                PL_put_list(term_t l);
        self.PL_put_list = self._lib.PL_put_list
        self.PL_put_list.argtypes = [term_t]
        self.PL_put_list.restype = None

        # PL_EXPORT(void)                PL_put_nil(term_t l);
        self.PL_put_nil = self._lib.PL_put_nil
        self.PL_put_nil.argtypes = [term_t]
        self.PL_put_nil.restype = None

        # PL_EXPORT(void)                PL_put_term(term_t t1, term_t t2);
        self.PL_put_term = self._lib.PL_put_term
        self.PL_put_term.argtypes = [term_t, term_t]
        self.PL_put_term.restype = None

        #                       /* construct a functor or list-cell */
        # PL_EXPORT(void)                PL_cons_functor(term_t h, functor_t f, ...);
        # class _PL_cons_functor(object):
        self.PL_cons_functor = self._lib.PL_cons_functor  # FIXME:

        # PL_EXPORT(void)                PL_cons_functor_v(term_t h, functor_t fd, term_t a0);
        self.PL_cons_functor_v = self._lib.PL_cons_functor_v
        self.PL_cons_functor_v.argtypes = [term_t, functor_t, term_t]
        self.PL_cons_functor_v.restype = None

        # PL_EXPORT(void)                PL_cons_list(term_t l, term_t h, term_t t);
        self.PL_cons_list = self._lib.PL_cons_list
        self.PL_cons_list.argtypes = [term_t, term_t, term_t]
        self.PL_cons_list.restype = None

        #
        # term_t PL_exception(qid_t qid)
        self.PL_exception = self._lib.PL_exception
        self.PL_exception.argtypes = [qid_t]
        self.PL_exception.restype = term_t
        #
        self.PL_register_foreign = self._lib.PL_register_foreign
        self.PL_register_foreign = check_strings(0, None)(self.PL_register_foreign)

        #
        # PL_EXPORT(atom_t)      PL_new_atom(const char *s);
        self.PL_new_atom = self._lib.PL_new_atom
        self.PL_new_atom.argtypes = [c_char_p]
        self.PL_new_atom.restype = atom_t

        self.PL_new_atom = check_strings(0, None)(self.PL_new_atom)

        # PL_EXPORT(functor_t)   PL_new_functor(atom_t f, int a);
        self.PL_new_functor = self._lib.PL_new_functor
        self.PL_new_functor.argtypes = [atom_t, c_int]
        self.PL_new_functor.restype = functor_t

        #                /*******************************
        #                *           COMPARE            *
        #                *******************************/
        #
        # PL_EXPORT(int)         PL_compare(term_t t1, term_t t2);
        # PL_EXPORT(int)         PL_same_compound(term_t t1, term_t t2);
        self.PL_compare = self._lib.PL_compare
        self.PL_compare.argtypes = [term_t, term_t]
        self.PL_compare.restype = c_int

        self.PL_same_compound = self._lib.PL_same_compound
        self.PL_same_compound.argtypes = [term_t, term_t]
        self.PL_same_compound.restype = c_int

        #                /*******************************
        #                *      RECORDED DATABASE       *
        #                *******************************/
        #
        # PL_EXPORT(record_t)    PL_record(term_t term);
        self.PL_record = self._lib.PL_record
        self.PL_record.argtypes = [term_t]
        self.PL_record.restype = record_t

        # PL_EXPORT(void)                PL_recorded(record_t record, term_t term);
        self.PL_recorded = self._lib.PL_recorded
        self.PL_recorded.argtypes = [record_t, term_t]
        self.PL_recorded.restype = None

        # PL_EXPORT(void)                PL_erase(record_t record);
        self.PL_erase = self._lib.PL_erase
        self.PL_erase.argtypes = [record_t]
        self.PL_erase.restype = None

        #
        # PL_EXPORT(char *)      PL_record_external(term_t t, size_t *size);
        # PL_EXPORT(int)         PL_recorded_external(const char *rec, term_t term);
        # PL_EXPORT(int)         PL_erase_external(char *rec);

        self.PL_new_module = self._lib.PL_new_module
        self.PL_new_module.argtypes = [atom_t]
        self.PL_new_module.restype = module_t

        self.PL_is_initialised = self._lib.PL_is_initialised

        # PL_EXPORT(IOSTREAM *)  Sopen_string(IOSTREAM *s, char *buf, size_t sz, const char *m);
        self.Sopen_string = self._lib.Sopen_string
        self.Sopen_string.argtypes = [POINTER(IOSTREAM), c_char_p, c_size_t, c_char_p]
        self.Sopen_string.restype = POINTER(IOSTREAM)

        # PL_EXPORT(int)         Sclose(IOSTREAM *s);
        self.Sclose = self._lib.Sclose
        self.Sclose.argtypes = [POINTER(IOSTREAM)]

        # PL_EXPORT(int)         PL_unify_stream(term_t t, IOSTREAM *s);
        self.PL_unify_stream = self._lib.PL_unify_stream
        self.PL_unify_stream.argtypes = [term_t, POINTER(IOSTREAM)]

        self.func_names = set()
        self.func = {}
        self.unify = None

    @staticmethod
    def unifier(arity, *args):
        assert arity == 2
        # if PL_is_variable(args[0]):
        #    args[0].unify(args[1])
        try:
            return {args[0].value: args[1].value}
        except AttributeError:
            return {args[0].value: args[1]}

    def __del__(self):
        # only do something if prolog has been initialised
        if self.PL_is_initialised(None, None):
            # clean up the prolog system using the caught exit code
            # if exit code is None, the program exits normally and we can use 0
            # instead.
            # TODO Prolog documentation says cleanup with code 0 may be interrupted
            # If the program has come to an end the prolog system should not
            # interfere with that. Therefore we may want to use 1 instead of 0.
            self.PL_cleanup(int(_hook.exit_code or 0))
            _isCleaned = True

    @staticmethod
    def _findSwiplPathFromFindLib():
        """
        This function resorts to ctype's find_library to find the path to the
        DLL. The biggest problem is that find_library does not give the path to the
        resource file.

        :returns:
            A path to the swipl SO/DLL or None if it is not found.

        :returns type:
            {str, None}
        """

        path = (find_library('swipl') or
                find_library('pl') or
                find_library('libswipl')) # This last one is for Windows
        return path


    @staticmethod
    def _findSwiplFromExec():
        """
        This function tries to use an executable on the path to find SWI-Prolog
        SO/DLL and the resource file.

        :returns:
            A tuple of (path to the swipl DLL, path to the resource file)

        :returns type:
            ({str, None}, {str, None})
        """

        platform = sys.platform[:3]

        fullName = None
        swiHome = None

        try: # try to get library path from swipl executable.

            # We may have pl or swipl as the executable
            try:
                cmd = Popen(['swipl', '--dump-runtime-variables'], stdout=PIPE)
            except OSError:
                cmd = Popen(['pl', '--dump-runtime-variables'], stdout=PIPE)
            ret = cmd.communicate()

            # Parse the output into a dictionary
            ret = ret[0].decode().replace(';', '').splitlines()
            ret = [line.split('=', 1) for line in ret]
            rtvars = dict((name, value[1:-1]) for name, value in ret) # [1:-1] gets
                                                                      # rid of the
                                                                      # quotes

            if rtvars['PLSHARED'] == 'no':
                raise ImportError('SWI-Prolog is not installed as a shared '
                                  'library.')
            else: # PLSHARED == 'yes'
                swiHome = rtvars['PLBASE']   # The environment is in PLBASE
                if not os.path.exists(swiHome):
                    swiHome = None

                # determine platform specific path
                if platform == "win":
                    dllName = rtvars['PLLIB'][:-4] + '.' + rtvars['PLSOEXT']
                    path = os.path.join(rtvars['PLBASE'], 'bin')
                    fullName = os.path.join(path, dllName)

                    if not os.path.exists(fullName):
                        fullName = None

                elif platform == "cyg":
                    # e.g. /usr/lib/pl-5.6.36/bin/i686-cygwin/cygpl.dll

                    dllName = 'cygpl.dll'
                    path = os.path.join(rtvars['PLBASE'], 'bin', rtvars['PLARCH'])
                    fullName = os.path.join(path, dllName)

                    if not os.path.exists(fullName):
                        fullName = None

                elif platform == "dar":
                    dllName = 'lib' + rtvars['PLLIB'][2:] + '.' + rtvars['PLSOEXT']
                    path = os.path.join(rtvars['PLBASE'], 'lib', rtvars['PLARCH'])
                    baseName = os.path.join(path, dllName)

                    if os.path.exists(baseName):
                        fullName = baseName
                    else:  # We will search for versions
                        fullName = None

                else: # assume UNIX-like
                    # The SO name in some linuxes is of the form libswipl.so.5.10.2,
                    # so we have to use glob to find the correct one
                    dllName = 'lib' + rtvars['PLLIB'][2:] + '.' + rtvars['PLSOEXT']
                    path = os.path.join(rtvars['PLBASE'], 'lib', rtvars['PLARCH'])
                    baseName = os.path.join(path, dllName)

                    if os.path.exists(baseName):
                        fullName = baseName
                    else:  # We will search for versions
                        pattern = baseName + '.*'
                        files = glob.glob(pattern)
                        if len(files) == 0:
                            fullName = None
                        elif len(files) == 1:
                            fullName = files[0]
                        else:  # Will this ever happen?
                            fullName = None

        except (OSError, KeyError): # KeyError from accessing rtvars
            pass

        return (fullName, swiHome)


    @staticmethod
    def _findSwiplWin():
        import re

        """
        This function uses several heuristics to gues where SWI-Prolog is installed
        in Windows. It always returns None as the path of the resource file because,
        in Windows, the way to find it is more robust so the SWI-Prolog DLL is
        always able to find it.
    
        :returns:
            A tuple of (path to the swipl DLL, path to the resource file)
    
        :returns type:
            ({str, None}, {str, None})
        """

        dllNames = ('swipl.dll', 'libswipl.dll')

        # First try: check the usual installation path (this is faster but
        # hardcoded)
        programFiles = os.getenv('ProgramFiles')
        paths = [os.path.join(programFiles, r'pl\bin', dllName)
                 for dllName in dllNames]
        for path in paths:
            if os.path.exists(path):
                return (path, None)

        # Second try: use the find_library
        path = SWIPl._findSwiplPathFromFindLib()
        if path is not None and os.path.exists(path):
            return (path, None)

        # Third try: use reg.exe to find the installation path in the registry
        # (reg should be installed in all Windows XPs)
        try:
            cmd = Popen(['reg', 'query',
                r'HKEY_LOCAL_MACHINE\Software\SWI\Prolog',
                '/v', 'home'], stdout=PIPE)
            ret = cmd.communicate()

            # Result is like:
            # ! REG.EXE VERSION 3.0
            #
            # HKEY_LOCAL_MACHINE\Software\SWI\Prolog
            #    home        REG_SZ  C:\Program Files\pl
            # (Note: spaces may be \t or spaces in the output)
            ret = ret[0].splitlines()
            ret = [line.decode("utf-8") for line in ret if len(line) > 0]
            pattern = re.compile('[^h]*home[^R]*REG_SZ( |\t)*(.*)$')
            match = pattern.match(ret[-1])
            if match is not None:
                path = match.group(2)

                paths = [os.path.join(path, 'bin', dllName)
                         for dllName in dllNames]
                for path in paths:
                    if os.path.exists(path):
                        return (path, None)

        except OSError:
            # reg.exe not found? Weird...
            pass

        # May the exec is on path?
        (path, swiHome) = SWIPl._findSwiplFromExec()
        if path is not None:
            return (path, swiHome)

        # Last try: maybe it is in the current dir
        for dllName in dllNames:
            if os.path.exists(dllName):
                return (dllName, None)

        return (None, None)

    @staticmethod
    def _findSwiplLin():
        """
        This function uses several heuristics to guess where SWI-Prolog is
        installed in Linuxes.

        :returns:
            A tuple of (path to the swipl so, path to the resource file)

        :returns type:
            ({str, None}, {str, None})
        """

        # Maybe the exec is on path?
        (path, swiHome) = SWIPl._findSwiplFromExec()
        if path is not None:
            return (path, swiHome)

        # If it is not, use  find_library
        path = SWIPl._findSwiplPathFromFindLib()
        if path is not None:
            return (path, swiHome)

        # Our last try: some hardcoded paths.
        paths = ['/lib', '/usr/lib', '/usr/local/lib', '.', './lib']
        names = ['libswipl.so', 'libpl.so']

        path = None
        for name in names:
            for try_ in paths:
                try_ = os.path.join(try_, name)
                if os.path.exists(try_):
                    path = try_
                    break

        if path is not None:
            return (path, swiHome)

        return (None, None)

    @staticmethod
    def _findSwiplMacOSHome():
        """
        This function is guesing where SWI-Prolog is
        installed in MacOS via .app.

        :parameters:
          -  `swi_ver` (str) - Version of SWI-Prolog in '[0-9].[0-9].[0-9]' format

        :returns:
            A tuple of (path to the swipl so, path to the resource file)

        :returns type:
            ({str, None}, {str, None})
        """

        # Need more help with MacOS
        # That way works, but need more work
        names = ['libswipl.dylib', 'libpl.dylib']

        path = os.environ.get('SWI_HOME_DIR')
        if path is None:
            path = os.environ.get('SWI_LIB_DIR')
            if path is None:
                path = os.environ.get('PLBASE')
                if path is None:
                    swi_ver = get_swi_ver()
                    path = '/Applications/SWI-Prolog.app/Contents/swipl-' + swi_ver + '/lib/'

        paths = [path]

        for name in names:
            for path in paths:
                (path_res, back_path) = walk(path, name)

                if path_res is not None:
                    os.environ['SWI_LIB_DIR'] = back_path
                    return (path_res, None)

        return (None, None)

    @staticmethod
    def _findSwiplDar():
        """
        This function uses several heuristics to guess where SWI-Prolog is
        installed in MacOS.

        :returns:
            A tuple of (path to the swipl so, path to the resource file)

        :returns type:
            ({str, None}, {str, None})
        """

        # If the exec is in path
        (path, swiHome) = SWIPl._findSwiplFromExec()
        if path is not None:
            return (path, swiHome)

        # If it is not, use  find_library
        path = SWIPl._findSwiplPathFromFindLib()
        if path is not None:
            return (path, swiHome)

        # Last guess, searching for the file
        paths = ['.', './lib', '/usr/lib/', '/usr/local/lib', '/opt/local/lib']
        names = ['libswipl.dylib', 'libpl.dylib']

        for name in names:
            for path in paths:
                path = os.path.join(path, name)
                if os.path.exists(path):
                    return (path, None)

        return (None, None)

    @staticmethod
    def _findSwipl():
        """
        This function makes a big effort to find the path to the SWI-Prolog shared
        library. Since this is both OS dependent and installation dependent, we may
        not aways succeed. If we do, we return a name/path that can be used by
        CDLL(). Otherwise we raise an exception.

        :return: Tuple. Fist element is the name or path to the library that can be
                 used by CDLL. Second element is the path were SWI-Prolog resource
                 file may be found (this is needed in some Linuxes)
        :rtype: Tuple of strings
        :raises ImportError: If we cannot guess the name of the library
        """

        # Now begins the guesswork
        platform = sys.platform[:3]
        if platform == "win":  # In Windows, we have the default installer
            # path and the registry to look
            (path, swiHome) = SWIPl._findSwiplWin()

        elif platform in ("lin", "cyg"):
            (path, swiHome) = SWIPl._findSwiplLin()

        elif platform == "dar":  # Help with MacOS is welcome!!
            (path, swiHome) = SWIPl._findSwiplDar()

            if path is None:
                (path, swiHome) = SWIPl._findSwiplMacOSHome()

        else:
            # This should work for other UNIX
            (path, swiHome) = SWIPl._findSwiplLin()

        # This is a catch all raise
        if path is None:
            raise ImportError('Could not find the SWI-Prolog library in this '
                              'platform. If you are sure it is installed, please '
                              'open an issue.')
        else:
            return (path, swiHome)


    @staticmethod
    def _fixWindowsPath(dll):
        """
        When the path to the DLL is not in Windows search path, Windows will not be
        able to find other DLLs on the same directory, so we have to add it to the
        path. This function takes care of it.

        :parameters:
          -  `dll` (str) - File name of the DLL
        """

        if sys.platform[:3] != 'win':
            return  # Nothing to do here

        pathToDll = os.path.dirname(dll)
        currentWindowsPath = os.getenv('PATH')

        if pathToDll not in currentWindowsPath:
            # We will prepend the path, to avoid conflicts between DLLs
            newPath = pathToDll + ';' + currentWindowsPath
            os.putenv('PATH', newPath)


def walk(path, name):
    """
    This function is a 2-time recursive func,
    that findin file in dirs
    
    :parameters:
      -  `path` (str) - Directory path
      -  `name` (str) - Name of file, that we lookin for
      
    :returns:
        Path to the swipl so, path to the resource file

    :returns type:
        (str)
    """
    back_path = path[:]
    path = os.path.join(path, name)
    
    if os.path.exists(path):
        return path
    else:
        for dir_ in os.listdir(back_path):
            path = os.path.join(back_path, dir_)

            if os.path.isdir(path):
                res_path = walk(path, name)
                if res_path is not None:
                    return (res_path, back_path)

    return None


def get_swi_ver():
    import re
    swi_ver = input(
                'Please enter you SWI-Prolog version in format "X.Y.Z": ')
    match = re.search(r'[0-9]\.[0-9]\.[0-9]')
    if match is None:
        raise InputError('Error, type normal version')
    
    return swi_ver


_stringMap = {}
def str_to_bytes(string):
    """
    Turns a string into a bytes if necessary (i.e. if it is not already a bytes
    object or None).
    If string is None, int or c_char_p it will be returned directly.

    :param string: The string that shall be transformed
    :type string: str, bytes or type(None)
    :return: Transformed string
    :rtype: c_char_p compatible object (bytes, c_char_p, int or None)
    """
    if string is None or isinstance(string, (int, c_char_p)):
        return string

    if not isinstance(string, bytes):
        if string not in _stringMap:
            _stringMap[string] = string.encode()
        string = _stringMap[string]

    return string

def list_to_bytes_list(strList):
    """
    This function turns an array of strings into a pointer array
    with pointers pointing to the encodings of those strings
    Possibly contained bytes are kept as they are.

    :param strList: List of strings that shall be converted
    :type strList: List of strings
    :returns: Pointer array with pointers pointing to bytes
    :raises: TypeError if strList is not list, set or tuple
    """
    pList = c_char_p * len(strList)

    # if strList is already a pointerarray or None, there is nothing to do
    if isinstance(strList, (pList, type(None))):
        return strList

    if not isinstance(strList, (list, set, tuple)):
        raise TypeError("strList must be list, set or tuple, not " +
                str(type(strList)))

    pList = pList()
    for i, elem in enumerate(strList):
        pList[i] = str_to_bytes(elem)
    return pList

# create a decorator that turns the incoming strings into c_char_p compatible
# butes or pointer arrays
def check_strings(strings, arrays):
    """
    Decorator function which can be used to automatically turn an incoming
    string into a bytes object and an incoming list to a pointer array if
    necessary.

    :param strings: Indices of the arguments must be pointers to bytes
    :type strings: List of integers
    :param arrays: Indices of the arguments must be arrays of pointers to bytes
    :type arrays: List of integers
    """

    # if given a single element, turn it into a list
    if isinstance(strings, int):
        strings = [strings]
    elif strings is None:
        strings = []

    # check if all entries are integers
    for i,k in enumerate(strings):
        if not isinstance(k, int):
            raise TypeError(('Wrong type for index at {0} '+
                    'in strings. Must be int, not {1}!').format(i,k))

    # if given a single element, turn it into a list
    if isinstance(arrays, int):
        arrays = [arrays]
    elif arrays is None:
        arrays = []

    # check if all entries are integers
    for i,k in enumerate(arrays):
        if not isinstance(k, int):
            raise TypeError(('Wrong type for index at {0} '+
                    'in arrays. Must be int, not {1}!').format(i,k))

    # check if some index occurs in both
    if set(strings).intersection(arrays):
        raise ValueError('One or more elements occur in both arrays and ' +
                ' strings. One parameter cannot be both list and string!')

    # create the checker that will check all arguments given by argsToCheck
    # and turn them into the right datatype.
    def checker(func):
        def check_and_call(*args):
            args = list(args)
            for i in strings:
                arg = args[i]
                args[i] = str_to_bytes(arg)
            for i in arrays:
                arg = args[i]
                args[i] = list_to_bytes_list(arg)

            return func(*args)

        return check_and_call

    return checker

# PySwip constants
PYSWIP_MAXSTR = 1024
c_int_p = c_void_p
c_long_p = c_void_p
c_double_p = c_void_p
c_uint_p = c_void_p

# constants (from SWI-Prolog.h)
# PL_unify_term() arguments
PL_VARIABLE = 1  # nothing
PL_ATOM = 2  # const char
PL_INTEGER = 3  # int
PL_FLOAT = 4  # double
PL_STRING = 5  # const char *
PL_TERM = 6  #
# PL_unify_term()
PL_FUNCTOR = 10  # functor_t, arg ...
PL_LIST = 11  # length, arg ...
PL_CHARS = 12  # const char *
PL_POINTER = 13  # void *
#               /* PlArg::PlArg(text, type) */
#define PL_CODE_LIST     (14)       /* [ascii...] */
#define PL_CHAR_LIST     (15)       /* [h,e,l,l,o] */
#define PL_BOOL      (16)       /* PL_set_feature() */
#define PL_FUNCTOR_CHARS (17)       /* PL_unify_term() */
#define _PL_PREDICATE_INDICATOR (18)    /* predicate_t (Procedure) */
#define PL_SHORT     (19)       /* short */
#define PL_INT       (20)       /* int */
#define PL_LONG      (21)       /* long */
#define PL_DOUBLE    (22)       /* double */
#define PL_NCHARS    (23)       /* unsigned, const char * */
#define PL_UTF8_CHARS    (24)       /* const char * */
#define PL_UTF8_STRING   (25)       /* const char * */
#define PL_INT64     (26)       /* int64_t */
#define PL_NUTF8_CHARS   (27)       /* unsigned, const char * */
#define PL_NUTF8_CODES   (29)       /* unsigned, const char * */
#define PL_NUTF8_STRING  (30)       /* unsigned, const char * */
#define PL_NWCHARS   (31)       /* unsigned, const wchar_t * */
#define PL_NWCODES   (32)       /* unsigned, const wchar_t * */
#define PL_NWSTRING  (33)       /* unsigned, const wchar_t * */
#define PL_MBCHARS   (34)       /* const char * */
#define PL_MBCODES   (35)       /* const char * */
#define PL_MBSTRING  (36)       /* const char * */

#       /********************************
#       * NON-DETERMINISTIC CALL/RETURN *
#       *********************************/
#
#  Note 1: Non-deterministic foreign functions may also use the deterministic
#    return methods PL_succeed and PL_fail.
#
#  Note 2: The argument to PL_retry is a 30 bits signed integer (long).

PL_FIRST_CALL = 0
PL_CUTTED = 1
PL_PRUNED = PL_CUTTED
PL_REDO = 2

PL_FA_NOTRACE = 0x01  # foreign cannot be traced
PL_FA_TRANSPARENT = 0x02  # foreign is module transparent
PL_FA_NONDETERMINISTIC  = 0x04  # foreign is non-deterministic
PL_FA_VARARGS = 0x08  # call using t0, ac, ctx
PL_FA_CREF = 0x10  # Internal: has clause-reference */

#        /*******************************
#        *       CALL-BACK      *
#        *******************************/

PL_Q_DEBUG = 0x01  # = TRUE for backward compatibility
PL_Q_NORMAL = 0x02  # normal usage
PL_Q_NODEBUG =  0x04  # use this one
PL_Q_CATCH_EXCEPTION = 0x08  # handle exceptions in C
PL_Q_PASS_EXCEPTION = 0x10  # pass to parent environment
PL_Q_DETERMINISTIC = 0x20  # call was deterministic

#        /*******************************
#        *         BLOBS        *
#        *******************************/

#define PL_BLOB_MAGIC_B 0x75293a00  /* Magic to validate a blob-type */
#define PL_BLOB_VERSION 1       /* Current version */
#define PL_BLOB_MAGIC   (PL_BLOB_MAGIC_B|PL_BLOB_VERSION)

#define PL_BLOB_UNIQUE  0x01        /* Blob content is unique */
#define PL_BLOB_TEXT    0x02        /* blob contains text */
#define PL_BLOB_NOCOPY  0x04        /* do not copy the data */
#define PL_BLOB_WCHAR   0x08        /* wide character string */

#        /*******************************
#        *      CHAR BUFFERS    *
#        *******************************/

CVT_ATOM = 0x0001
CVT_STRING = 0x0002
CVT_LIST = 0x0004
CVT_INTEGER = 0x0008
CVT_FLOAT = 0x0010
CVT_VARIABLE = 0x0020
CVT_NUMBER = CVT_INTEGER | CVT_FLOAT
CVT_ATOMIC = CVT_NUMBER | CVT_ATOM | CVT_STRING
CVT_WRITE = 0x0040  # as of version 3.2.10
CVT_ALL = CVT_ATOMIC | CVT_LIST
CVT_MASK = 0x00ff

BUF_DISCARDABLE = 0x0000
BUF_RING = 0x0100
BUF_MALLOC = 0x0200

CVT_EXCEPTION = 0x10000  # throw exception on error

argv = list_to_bytes_list(sys.argv + [None])
argc = len(sys.argv)

#                  /*******************************
#                  *             TYPES            *
#                  *******************************/
#
# typedef uintptr_t       atom_t;         /* Prolog atom */
# typedef uintptr_t       functor_t;      /* Name/arity pair */
# typedef void *          module_t;       /* Prolog module */
# typedef void *          predicate_t;    /* Prolog procedure */
# typedef void *          record_t;       /* Prolog recorded term */
# typedef uintptr_t       term_t;         /* opaque term handle */
# typedef uintptr_t       qid_t;          /* opaque query handle */
# typedef uintptr_t       PL_fid_t;       /* opaque foreign context handle */
# typedef void *          control_t;      /* non-deterministic control arg */
# typedef void *          PL_engine_t;    /* opaque engine handle */
# typedef uintptr_t       PL_atomic_t;    /* same a word */
# typedef uintptr_t       foreign_t;      /* return type of foreign functions */
# typedef wchar_t         pl_wchar_t;     /* Prolog wide character */
# typedef foreign_t       (*pl_function_t)(); /* foreign language functions */

atom_t = c_uint_p
functor_t = c_uint_p
module_t = c_void_p
predicate_t = c_void_p
record_t = c_void_p
term_t = c_uint_p
qid_t = c_uint_p
PL_fid_t = c_uint_p
fid_t = c_uint_p
control_t = c_void_p
PL_engine_t = c_void_p
PL_atomic_t = c_uint_p
foreign_t = c_uint_p
pl_wchar_t = c_wchar
intptr_t = c_long
ssize_t = intptr_t
wint_t = c_uint

#typedef struct
#{
#  int __count;
#  union
#  {
#    wint_t __wch;
#    char __wchb[4];
#  } __value;            /* Value so far.  */
#} __mbstate_t;

class _mbstate_t_value(Union):
    _fields_ = [("__wch",wint_t),
                ("__wchb",c_char*4)]

class mbstate_t(Structure):
    _fields_ = [("__count",c_int),
                ("__value",_mbstate_t_value)]

# stream related funcs
Sread_function = CFUNCTYPE(ssize_t, c_void_p, c_char_p, c_size_t)
Swrite_function = CFUNCTYPE(ssize_t, c_void_p, c_char_p, c_size_t)
Sseek_function = CFUNCTYPE(c_long, c_void_p, c_long, c_int)
Sseek64_function = CFUNCTYPE(c_int64, c_void_p, c_int64, c_int)
Sclose_function = CFUNCTYPE(c_int, c_void_p)
Scontrol_function = CFUNCTYPE(c_int, c_void_p, c_int, c_void_p)

# IOLOCK
IOLOCK = c_void_p

# IOFUNCTIONS
class IOFUNCTIONS(Structure):
    _fields_ = [("read",Sread_function),
                ("write",Swrite_function),
                ("seek",Sseek_function),
                ("close",Sclose_function),
                ("seek64",Sseek64_function),
                ("reserved",intptr_t*2)]

# IOENC
ENC_UNKNOWN,ENC_OCTET,ENC_ASCII,ENC_ISO_LATIN_1,ENC_ANSI,ENC_UTF8,ENC_UNICODE_BE,ENC_UNICODE_LE,ENC_WCHAR = tuple(range(9))
IOENC = c_int

# IOPOS
class IOPOS(Structure):
    _fields_ = [("byteno",c_int64),
                ("charno",c_int64),
                ("lineno",c_int),
                ("linepos",c_int),
                ("reserved", intptr_t*2)]

# IOSTREAM
class IOSTREAM(Structure):
    _fields_ = [("bufp",c_char_p),
                ("limitp",c_char_p),
                ("buffer",c_char_p),
                ("unbuffer",c_char_p),
                ("lastc",c_int),
                ("magic",c_int),
                ("bufsize",c_int),
                ("flags",c_int),
                ("posbuf",IOPOS),
                ("position",POINTER(IOPOS)),
                ("handle",c_void_p),
                ("functions",IOFUNCTIONS),
                ("locks",c_int),
                ("mutex",IOLOCK),
                ("closure_hook",CFUNCTYPE(None, c_void_p)),
                ("closure",c_void_p),
                ("timeout",c_int),
                ("message",c_char_p),
                ("encoding",IOENC)]
IOSTREAM._fields_.extend([("tee",IOSTREAM),
                ("mbstate",POINTER(mbstate_t)),
                ("reserved",intptr_t*6)])


#create an exit hook which captures the exit code for our cleanup function
class ExitHook(object):
    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

_hook = ExitHook()
_hook.hook()

_isCleaned = False
#create a property for Atom's delete method in order to avoid segmentation fault
cleaned = property(_isCleaned)
