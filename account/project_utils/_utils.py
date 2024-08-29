import importlib
from itertools import islice


def create_module(module_str):
    tmpss = module_str.split(',')
    assert len(tmpss) == 2, 'Error format of the module path: {}'.format(module_str)
    module_name, function_name = tmpss[0].strip(), tmpss[1].strip()
    somemodule = importlib.import_module(module_name, __package__)
    function = getattr(somemodule, function_name)
    return function


def chunked_iterable(iterable, size):
    it = iter(iterable)
    for first in it:
        yield [first] + list(islice(it, size - 1))

