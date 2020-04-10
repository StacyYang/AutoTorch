import autotorch as at

@at.obj(
    name=at.space.Choice('auto', 'torch'),
)
class myobj:
    def __init__(self, name):
        self.name = name

@at.func(
    framework=at.space.Choice('mxnet', 'pytorch'),
)
def myfunc(framework):
    return framework

@at.args(
    a=at.space.Real(1e-3, 1e-2, log=True),
    b=at.space.Real(1e-3, 1e-2),
    c=at.space.Int(1, 10),
    d=at.space.Choice('a', 'b', 'c', 'd'),
    e=at.space.Bool(),
    f=at.space.List(
            at.space.Int(1, 2),
            at.space.Choice(4, 5),
        ),
    g=at.space.Dict(
            a=at.Real(0, 10),
            obj=myobj(),
        ),
    h=at.space.Choice('test', myobj()),
    i = myfunc(),
    )
def train_fn(args, reporter):
    a, b, c, d, e, f, g, h, i = args.a, args.b, args.c, args.d, args.e, \
            args.f, args.g, args.h, args.i
    assert a <= 1e-2 and a >= 1e-3
    assert b <= 1e-2 and b >= 1e-3
    assert c <= 10 and c >= 1
    assert d in ['a', 'b', 'c', 'd']
    assert e in [True, False]
    assert f[0] in [1, 2]
    assert f[1] in [4, 5]
    assert g['a'] <= 10 and g['a'] >= 0
    assert g.obj.name in ['auto', 'torch']
    assert hasattr(h, 'name') or h == 'test'
    assert i in ['mxnet', 'pytorch']
    reporter(epoch=e, accuracy=0)

def test_fifo_scheduler():
    scheduler = at.scheduler.FIFOScheduler(train_fn,
                                           resource={'num_cpus': 2, 'num_gpus': 0},
                                           num_trials=20,
                                           reward_attr='accuracy',
                                           time_attr='epoch')
    scheduler.run()
    scheduler.join_jobs()
