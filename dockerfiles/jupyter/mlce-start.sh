file_path="${HOME}/.intdata"

# .intdata indicates if we extracted the home tar or not
if [ -f "$file_path" ]; then
    echo "Base data exists."
else
    # Perform actions when the file does not exist
    echo "Base data does not exist, extracting home backup..."
    cd / ; tar -xvf /tmp/basehome.tar
    echo "1" > $file_path
fi

cd ${HOME}

start-notebook.sh \
  --ip=0.0.0.0 \
  --port=8888 \
  --NotebookApp.token="" \
  --NotebookApp.password="" \
  --NotebookApp.allow_origin="*" \
  --NotebookApp.default_url=/lab
