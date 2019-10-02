class Transform(object):
    def __init__(self, name='Transform'):
        self.name = name

    def eval(self, x, *args, **kwargs):
        self.__eval__(*args, **kwargs)
        return x

    def __eval__(self, x, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
