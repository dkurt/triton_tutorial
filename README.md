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

## Home Work

1) Choose one of any lightweight model (or at least one your PC can handle) from the Hub: https://huggingface.co/ . The model should process any of media data (text, audio, voice, image, video) and return any human-understandable results. In example:
  * Image classification with object class name
  * Generative models: text to image/video/audio/text
  * Object detection with image and bounding boxes on it
  * Data processing with text output (not embeddings or any other intermediate representations)
  * speech-to-text with audio as input and text on output
  * Noise reduction
  * Text summarization or question-answering
  * Any models applied on image but for video on any frame or for every n-th frame and interpolation
2) Create and host Triton pipeline locally that implements choosen pipilene end-to-end. That means media data as input and final result as output - with no post-processing on client side
3) Create and publish a Telegram bot (with Python backend) that accepts one of the possible media formats from user - sends it to Triton server and return back to user in chat.
