from mlrun.builder import upload_tarball, build

inline = """
print(1+1)
"""

tst = 2

# various build examples, need to run inside Kubernetes/jupyter
# or place the Kubernetes config file for the cluster in the default location

if tst == 1:
    build('yhaviv/ktests2:latest',
          requirements=['pandas'],
          inline_code=inline)

if tst == 2:
    upload_tarball('../buildtst', 'v3ios:///users/admin/context/src.tar.gz')
    build('yhaviv/ktests3:latest',
          source='/users/admin/context/src.tar.gz',
          requirements='requirements.txt')

if tst == 3:
    build('yhaviv/ktests3:latest',
          source='git://github.com/hodisr/iguazio_example.git',
          commands=['python setup.py install'])
