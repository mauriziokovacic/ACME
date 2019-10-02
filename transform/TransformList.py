from .Transform import *


class TransformList(list, Transform):
    def __init__(self, *transforms, name='TransformList'):
        for t in transforms:
            assert isinstance(t, Transform), 'Expected objects of type Transform. Got {}'.format(type(t))
        list.__init__(self, transforms)
        Transform.__init__(self, name=name)

    def __eval__(self, x, *args, **kwargs):
        for t in self:
            t.eval(x, *args, **kwargs)

    def append(self, item):
        assert isinstance(item, Transform), 'Expected item to be type Transform. Got {}'.format(type(item))
        return list.append(self, item)

    def insert(self, pos, item):
        assert isinstance(item, Transform), 'Expected item to be type Transform. Got {}'.format(type(item))
        return list.insert(self, pos, item)

    def extend(self, items):
        for item in items:
            assert isinstance(item, Transform), 'Expected item to be type Transform. Got {}'.format(type(item))
        return list.extend(self, items)
