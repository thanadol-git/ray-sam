# To build the docker image
```
docker build -f Dockerfile -t raysam .
```

# To run the docker image
```
docker run -it --shm-size 60G --name raysam --gpus all -v /storage/jingyug/:/storage/raysam_user/ raysam /bin/bash
```

NOTE: Replace /storage/jingyug/ with your own path