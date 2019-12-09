from .Transform import *


class TransformList(Transform, list):
    def __init__(self, *transforms):
        for t in transforms:
            assert isinstance(t, Transform), 'Expected objects of type Transform. Got {} instead.'.format(type(t))
        Transform.__init__(self)
        list.__init__(self, transforms)

    def __eval__(self, x, *args, **kwargs):
        for t in self:
            t.eval(x, *args, **kwargs)

    def append(self, item):
        assert isinstance(item, Transform), 'Expected item to be type Transform. Got {} instead.'.format(type(item))
        return list.append(self, item)

    def insert(self, pos, item):
        assert isinstance(item, Transform), 'Expected item to be type Transform. Got {} instead.'.format(type(item))
        return list.insert(self, pos, item)

    def extend(self, items):
        for item in items:
            assert isinstance(item, Transform), 'Expected item to be type Transform. Got {} instead.'.format(type(item))
        return list.extend(self, items)

    def extra_repr(self):
        text = '[\n'
        for i, t in enumerate(self):
            text += '\t({}) : {}\n'.format(i, str(t))
        text += ']'
        return text
