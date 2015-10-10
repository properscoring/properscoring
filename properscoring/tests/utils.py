import warnings
import contextlib


@contextlib.contextmanager
def suppress_warnings(msg=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', msg)
        yield
