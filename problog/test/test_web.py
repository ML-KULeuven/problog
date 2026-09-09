import unittest

try:
    # problog.web.server uses the POSIX-only `resource` module, so these tests
    # cannot even be imported on Windows.
    import resource  # noqa: F401
except ImportError:
    raise unittest.SkipTest("problog.web requires the POSIX 'resource' module")

from problog.web import server, server_debug


class TestWeb(unittest.TestCase):
    def test_web(self):
        server.main(["--local", "--test"])


if __name__ == "__main__":
    unittest.main()
