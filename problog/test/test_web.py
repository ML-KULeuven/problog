import unittest

from problog.web import server, server_debug


class TestWeb(unittest.TestCase):
    def test_web(self):
        server.main(["--local", "--test"])


if __name__ == "__main__":
    unittest.main()
