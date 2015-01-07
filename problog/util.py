from __future__ import print_function

import logging, time


class Timer(object) :

    def __init__(self, msg) :
        self.message = msg
        self.start_time = None

    def __enter__(self) :
        self.start_time = time.time()

    def __exit__(self, *args) :
        logger = logging.getLogger('problog')
        logger.info('%s: %.4fs' % (self.message, time.time()-self.start_time))

