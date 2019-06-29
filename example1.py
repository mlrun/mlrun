from mlrun import get_or_create_ctx

def my_func(ctx):
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(ctx.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(ctx.get_object('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    for i in range(1,4):
        ctx.log_metric('loss', 2*i, i)
    ctx.log_artifact('test.txt', body=b'abc is 123')


if __name__ == "__main__":
    ex = get_or_create_ctx('mytask')
    my_func(ex)
    ex.commit('aa')
