from mlrun import remote_run

url = 'http://54.93.111.58:30700'

remote_run(url, spec = {'spec': {'parameters':{'p1':8}}})
