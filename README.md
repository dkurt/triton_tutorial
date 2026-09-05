## Run model server
```
docker compose up --build
```

Then you may run tests
```
pytest run tests
```

## Model analyzer
1. Copy test video to `model_analyzer/inputs` with name `video` and no extension.
2. Launch server as usual `docker compose up`
3. Run `bash run_model_analyzer.sh`

## Tasks
### Non-Maximum Suppression

Apply NMS on `vehicle_detector` results for pipeline optimization to reduce number of not interesting bounding boxes

### TBD
