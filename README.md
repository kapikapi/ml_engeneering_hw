Build the container
```commandline
docker build -t docker-jupyter --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```
Run the container:
```commandline
docker run -p 8888:8888 -v *path_to_data_folder*:/src/static/ docker-jupyter 
```

Folder with data (*path_to_data_folder*) is expected to have .csv files in a folder called "data".

Metrics file will be saved to *path_to_data_folder*.