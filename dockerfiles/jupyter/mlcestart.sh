file_path="${HOME}/.intdata"

if [ -f "$file_path" ]; then
    echo "base data exists."
    # Perform actions when the file exists
else
    echo "base data does not exist."
    echo "1" > $file_path
    echo "base data exists."
    cd / ; tar -xvf /tmp/basehome.tar
    # Perform actions when the file does not exist
fi

cd ${HOME}

start-notebook.sh  --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --NotebookApp.password="" --NotebookApp.allow_origin="*" --NotebookApp.default_url=/lab
