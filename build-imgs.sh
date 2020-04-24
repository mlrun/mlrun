if [ $# -eq 0 ]
  then
    echo "No argument (ver tag) supplied"
fi

sudo docker build . -f ./dockerfiles/mlrun-api/Dockerfile -t mlrun/mlrun-api:$1
sudo docker push mlrun/mlrun-api:$1

sudo docker build . -t ./mlrun/mlrun:$1
sudo docker push mlrun/mlrun:$1

sudo docker build . -f ./dockerfiles/dask/Dockerfile -t mlrun/dask:$1
sudo docker push mlrun/dask:$1
