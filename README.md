## Run model server
```
docker compose up --build
```

Then you may run tests
```
pytest run tests
```

## Model analyzer
1. Download [test video](https://drive.google.com/file/d/14oplAtMYRqmGexvrG72q46b6Rytrgwhk/view?usp=drive_link)
2. Cut to 1 second sample: `ffmpeg -i cars_video.mp4 -ss 4 -to 5 -c copy model_analyzer/inputs/video.mp4`
3. Rename `video.mp4` to just `video` without extension.
4. Launch server as usual `docker compose up`
5. Run `bash run_model_analyzer.sh`

## Tasks
### Non-Maximum Suppression

Apply NMS on `vehicle_detector` results for pipeline optimization to reduce number of not interesting bounding boxes

### TBD
