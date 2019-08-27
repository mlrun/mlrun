from mlrun.builder import upload_tarball, build_image

inline = """
print(1+1)
"""

tst = 1

# various build examples, need to run inside Kubernetes/jupyter
# or place the Kubernetes config file for the cluster in the default location

if tst == 1:
    build_image('yhaviv/ktests2:latest',
          requirements=['pandas'],
          with_mlrun=False,
          inline_code=inline)

if tst == 2:
    upload_tarball('./', 'v3ios:///users/admin/context/src.tar.gz')
    build_image('yhaviv/ktests3:latest',
          source='/users/admin/context/src.tar.gz',
          requirements='requirements.txt')

if tst == 3:
    build_image('yhaviv/ktests3:latest',
          source='git://github.com/hodisr/iguazio_example.git',
          commands=['python setup.py install'])

if tst == 4:
    build_image('yhaviv/ktests3:latest',
          base_image='python:3.6',
          source='git://github.com/hodisr/iguazio_example.git',
          commands=['python setup.py install'])