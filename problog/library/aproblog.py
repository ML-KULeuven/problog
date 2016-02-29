from __future__ import print_function

from problog.extern import problog_export
from problog.evaluator import Semiring


@problog_export('+term', '+term', '+term', '+term', '+term', '+str', '+str')
def use_semiring(plus, times, zero, one, neg, is_dsp, is_nsp):

    class CustomSemiring(Semiring):
        pass

    CustomSemiring.zero = problog_export.database.create_function(zero, 1)
    CustomSemiring.one = problog_export.database.create_function(one, 1)
    CustomSemiring.plus = problog_export.database.create_function(plus, 3)
    CustomSemiring.times = problog_export.database.create_function(times, 3)
    CustomSemiring.pos_value = lambda s, x, k: x
    CustomSemiring.neg_value = problog_export.database.create_function(neg, 2)
    CustomSemiring.is_dsp = lambda s: is_dsp == 'true'
    CustomSemiring.is_nsp = lambda s: is_nsp == 'true'

    semiring = CustomSemiring()
    problog_export.database.set_data('semiring', semiring)

    return ()


@problog_export('+term', '+term', '+term', '+term', '+term', '+term', '+str', '+str')
def use_semiring(plus, times, zero, one, pos, neg, is_dsp, is_nsp):

    class CustomSemiring(Semiring):
        pass

    CustomSemiring.zero = problog_export.database.create_function(zero, 1)
    CustomSemiring.one = problog_export.database.create_function(one, 1)
    CustomSemiring.plus = problog_export.database.create_function(plus, 3)
    CustomSemiring.times = problog_export.database.create_function(times, 3)
    CustomSemiring.pos_value = problog_export.database.create_function(pos, 3)
    CustomSemiring.neg_value = problog_export.database.create_function(neg, 3)
    CustomSemiring.is_dsp = lambda s: is_dsp == 'true'
    CustomSemiring.is_nsp = lambda s: is_nsp == 'true'

    semiring = CustomSemiring()
    problog_export.database.set_data('semiring', semiring)

    return ()


