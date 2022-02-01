Firstly build the image:
```
docker build -t airflow_d:latest .
```
And then
```
docker-compose up
```
Go to http://0.0.0.0:8080/admin/airflow/ to see the UI and trigger "recs" DAG.