### Welcome to scam detector repo

This repo have action setup to image is pushed to dockerhub. 

# run prebuild docker image To run same
1. `docker pull mbaadror/embedded-system-models:latest`
2. `docker run --rm -it  -p 8000:8000 -t mbaadror/embedded-system-models:latest`



#### How to setup locally
1. to create docker image locally `bash | ./local/docker_commands/build_image.sh`
2. run docker image `bash | ./local/docker_commands/run_image.sh`
