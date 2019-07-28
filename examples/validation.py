from mlrun import get_or_create_ctx

def my_job():
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    context = get_or_create_ctx('validation')
    
    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print('file\n{}\n'.format(context.get_object('model.txt').get()))
    
    context.log_artifact('validation.html', body=b'<b> validated <b>', viewer='web-app')

if __name__ == "__main__":
    my_job()