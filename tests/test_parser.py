from mlrun import run

tests = [('ls', '', [None, 'ls', None, []]),
         ('ls -l x', '', [None, 'ls', None, ['-l', 'x']]),
         ('job://alpine#ls -l', 'job', ['alpine', 'ls', None, ['-l']]),
         ('job://alpine#test.py', 'job', ['alpine', 'test.py', '', []]),
         ('test.py', '', [None, 'test.py', '', []]),
         (r'C:\test.py', '', [None, r'C:\test.py', '', []]),
         ('http://my.url:5000/func', 'remote', 'http://my.url:5000/func'),
         ]

def test_cmd_parse():
    for t in tests:
        cmd, kind, params = t
        newkind, r = run.process_runtime(cmd, {})
        assert kind == newkind, 'error parsing kind in {} kind=()'.format(cmd, newkind)

        if kind == 'remote':
            assert r.get('command') == params, 'url parsing error {}'.format(params)
        else:
            assert r.get('image') == params[0], 'expected image: {}'.format(params[0])
            assert r.get('command') == params[1], 'expected command: {}'.format(params[1])
            assert r.get('args') == params[3], 'expected args: {}'.format(params[3])
        print(cmd, r)

def test_cmd_parse2():
    for t in tests:
        cmd, kind, params = t
        runtime = {'command': cmd}
        newkind, r = run.process_runtime('', runtime)
        assert kind == newkind, 'error parsing kind in {} kind=()'.format(cmd, newkind)

        if kind == 'remote':
            assert r.get('command') == params, 'url parsing error {}'.format(params)
        else:
            assert r.get('image') == params[0], 'expected image: {}'.format(params[0])
            assert r.get('command') == params[1], 'expected command: {}'.format(params[1])
            #assert r.get('handler') == params[2], 'expected image: {}'.format(params[2])
            assert r.get('args') == params[3], 'expected args: {}'.format(params[3])
        print(cmd, r)

