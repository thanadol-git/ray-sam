# To build the docker image
```
docker build -f Dockerfile -t raysam --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USER=$(whoami) --build-arg GROUP=$(id -g -n) .
```

# To run the docker image
```
docker run -it --shm-size 60G --name raysam --gpus all -v /storage/jingyug/:/storage/jingyug/ raysam /bin/bash
```

NOTE: Replace /storage/jingyug/ with your own path