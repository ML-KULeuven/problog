"""
Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import contextlib
import signal
import unittest
from unittest import mock

from problog import forward
from problog import get_evaluatable
from problog.evaluator import SemiringProbability
from problog.program import PrologString

# noinspection PyBroadException
try:
    from pysdd import sdd  # noqa: F401

    has_sdd = True
except Exception:
    has_sdd = False


@unittest.skipUnless(has_sdd, "ForwardSDD compiles with pysdd")
class TestForwardWithoutAlarm(unittest.TestCase):
    """signal.alarm is POSIX only, and build_dd cancelled an alarm it had not
    necessarily set, so every fsdd compilation raised AttributeError on Windows
    whether a timeout was asked for or not -- see ML-KULeuven/problog#151.

    HAS_ALARM stands in for the platform here, so these run everywhere.
    """

    program = "0.5::a. 0.5::b. c :- a, b. query(c)."

    @contextlib.contextmanager
    def without_alarm(self):
        """Stand in for Windows: no signal.alarm, and none to be found."""
        with mock.patch.object(forward, "HAS_ALARM", False):
            # On Windows there is nothing to take away to begin with.
            alarm = getattr(signal, "alarm", None)
            if alarm is not None:
                del signal.alarm
            try:
                yield
            finally:
                if alarm is not None:
                    signal.alarm = alarm

    def evaluate(self, **kwdargs):
        kc = get_evaluatable(name="fsdd").create_from(
            PrologString(self.program), **kwdargs
        )
        computed = kc.evaluate(semiring=SemiringProbability())
        return {str(k): v for k, v in computed.items()}

    def test_fsdd_compiles(self):
        self.assertAlmostEqual(0.25, self.evaluate()["c"])

    def test_fsdd_compiles_without_alarm(self):
        with self.without_alarm():
            self.assertAlmostEqual(0.25, self.evaluate()["c"])

    def test_timeout_without_alarm_is_reported_not_raised(self):
        """Asking for a timeout where none can be set should say so and go on."""
        with self.without_alarm():
            with self.assertLogs("problog", level="WARNING") as logged:
                self.assertAlmostEqual(0.25, self.evaluate(compile_timeout=10)["c"])
        self.assertIn("signal.alarm", "".join(logged.output))
