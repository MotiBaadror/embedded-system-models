### Welcome to scam detector repo

This repo have action setup to image is pushed to dockerhub. 

# run prebuild docker image To run same
1. `docker pull mbaadror/embedded-system-models:latest`
2. `docker run --rm -it  -p 8000:8000 -t mbaadror/embedded-system-models:latest`
3. send post request to url :`http://localhost:8000/predict` with  `data = {'text': 'WINNER!! As a valued network customer you have been selected to receivea '}`
4. or request can be send using the script `./local/flask_app_client.py`

#### How to setup locally
1. to create docker image locally `bash | ./local/docker_commands/build_image.sh`
2. run docker image `bash | ./local/docker_commands/run_image.sh`

#### Results
###### Loss and Accuracy Curve 
![Image](./resources/loss_curve.png)

###### Sample Result 
![image](./resources/sample_result.png)