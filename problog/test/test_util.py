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
import signal
import time
import unittest
from unittest import mock

from problog import util
from problog.errors import process_error


class TestGlobalTimer(unittest.TestCase):
    """--timeout was implemented with signal.alarm, which is POSIX only, so
    'problog --timeout N' raised AttributeError on Windows instead of timing
    anything -- see ML-KULeuven/problog#151.

    HAS_ALARM stands in for the platform, so both paths run everywhere.
    """

    def setUp(self):
        self.addCleanup(util.stop_timer)

    def wait(self, seconds):
        """Spin, so the interrupt lands on a bytecode boundary either way."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.01)

    def test_timer_interrupts_without_alarm(self):
        with mock.patch.object(util, "HAS_ALARM", False):
            util.start_timer(1)
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.wait(10)
            util.stop_timer()
        # errors.py tells a timeout from a Ctrl+C by the message.
        self.assertIn("Timeout", str(caught.exception))
        self.assertEqual("Timeout exceeded", process_error(caught.exception))

    @unittest.skipUnless(util.HAS_ALARM, "signal.alarm is POSIX only")
    def test_timer_interrupts_with_alarm(self):
        """The path every current user is on, unchanged."""
        util.start_timer(1)
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.wait(10)
        util.stop_timer()
        self.assertEqual("Timeout exceeded", process_error(caught.exception))

    def test_stopped_timer_does_not_interrupt_without_alarm(self):
        with mock.patch.object(util, "HAS_ALARM", False):
            util.start_timer(1)
            util.stop_timer()
            self.wait(1.5)  # would have fired by now

    def test_stopping_restores_the_interrupt_handler(self):
        before = signal.getsignal(signal.SIGINT)
        with mock.patch.object(util, "HAS_ALARM", False):
            util.start_timer(60)
            self.assertIsNot(before, signal.getsignal(signal.SIGINT))
            util.stop_timer()
        self.assertIs(before, signal.getsignal(signal.SIGINT))

    def test_a_real_interrupt_is_not_reported_as_a_timeout(self):
        """Ctrl+C while the timer is pending is still the user's interrupt."""
        with mock.patch.object(util, "HAS_ALARM", False):
            util.start_timer(60)
            handler = signal.getsignal(signal.SIGINT)
            with self.assertRaises(KeyboardInterrupt) as caught:
                handler(signal.SIGINT, None)
            util.stop_timer()
        self.assertNotIn("Timeout", str(caught.exception))
        self.assertEqual("Interrupted by user", process_error(caught.exception))

    def test_no_timeout_starts_nothing(self):
        with mock.patch.object(util, "HAS_ALARM", False):
            before = signal.getsignal(signal.SIGINT)
            util.start_timer(0)
            self.assertIs(before, signal.getsignal(signal.SIGINT))
