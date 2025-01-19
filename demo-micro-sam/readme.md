# Introduction

This is a demo for the micro SAM model. You can follow try running the demo code in this folder to see how the micro SAM model trains and predicts.

# To build the docker image
```
docker build -f Dockerfile -t raysam .
```

# To run the docker image
```
docker run -it --shm-size 60G --name raysam --gpus all -v /storage/jingyug/:/storage/raysam_user/ raysam /bin/bash
```

NOTE: Replace /storage/jingyug/ with your own path