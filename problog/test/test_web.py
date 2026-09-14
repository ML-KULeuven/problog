import json
import tempfile
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

    def test_mpe_request(self):
        """Every request failed before ProbLog even started.

        The server limits the memory of the ProbLog process it starts to
        --memout gigabytes, a float, and passed that product to setrlimit.
        Python 3.10 stopped accepting floats there, so the limit could not be
        set and the subprocess module raised before running anything.

        MPE rather than plain inference, so the request also runs maxsatz the
        way the server does.
        """
        model = "\n".join(
            [
                "0.7::burglary.",
                "0.2::earthquake.",
                "0.9::alarm :- burglary, earthquake.",
                "0.8::alarm :- burglary, \\+earthquake.",
                "0.1::alarm :- \\+burglary, earthquake.",
                "evidence(alarm).",
            ]
        )
        # Dispatch the way the request handler does; handle_url registers the
        # handlers but does not return them.
        handler = server.PATHS[server.api_root + "mpe"]
        cache_dir = server.CACHE_DIR
        with tempfile.TemporaryDirectory() as tmp:
            server.CACHE_DIR = tmp
            try:
                code, _, body = handler(model=[model], callback=["cb"])
            finally:
                server.CACHE_DIR = cache_dir

        self.assertEqual(200, code)
        self.assertTrue(body.startswith("cb(") and body.endswith(");"), body)
        result = json.loads(body[len("cb(") : -len(");")])
        self.assertTrue(result["SUCCESS"], result)
        self.assertIn(["burglary", True], result["atoms"])
        self.assertIn(["earthquake", False], result["atoms"])


if __name__ == "__main__":
    unittest.main()
