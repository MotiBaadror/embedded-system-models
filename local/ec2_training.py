from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import boto3

from pytorch_model_framework.io_strategy import S3Strategy
from local.training_struct import TrainingBasePaths
from models.resnet.resnet_full_config import ResnetFullConfig

@dataclass
class TrainingExecutionDetails:
    job_definition: str
    command: List[str]


TRAINING_TYPE_MAP = {
    ResnetFullConfig: TrainingExecutionDetails(
        job_definition="ec2-ecs-cpu-job-definition:1",
        command=["/workspace/app/entrypoint_train_mvit.sh"],
    )
}


def tell_user_how_to_trigger_run_with_config(location: str):
    print(f"After you SSH into the instance, start your training with the following commands:")
    print(
        "aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 654654419588.dkr.ecr.us-east-2.amazonaws.com"
    )
    print(
        f'docker run -d --gpus all --shm-size 16gb -e S3_PATH_TO_CONFIG="{location}" -v /home/ec2-user/data/:/workspace/app/data/input/ --name game-video-filter-repo 654654419588.dkr.ecr.us-east-2.amazonaws.com/game-video-filter-repo:latest'
    )


def upload_file_to_s3(
    contents: str, training_location: TrainingBasePaths, experiment_descriptor: str, job_name: Optional[str] = None
):
    location = f"{training_location.output_path}/{experiment_descriptor}"
    s3 = S3Strategy(location)
    file_name = "input_config.json" if not job_name else f"{job_name}.json"
    s3.save_file(file_name=file_name, contents=contents)
    return f"{location}/{file_name}"


def handle_instance_create_config(config_json: str, training_location: TrainingBasePaths, experiment_descriptor: str):
    location = upload_file_to_s3(
        contents=config_json, training_location=training_location, experiment_descriptor=experiment_descriptor
    )

    tell_user_how_to_trigger_run_with_config(location=location)


def handle_batch_create_config(
    config_json: str, training_location: TrainingBasePaths, experiment_descriptor: str, training_type
):
    job_name = f'{experiment_descriptor}_{datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")}'
    location = upload_file_to_s3(
        contents=config_json,
        training_location=training_location,
        experiment_descriptor=experiment_descriptor,
        job_name=job_name,
    )

    print(f"Submitting job_name: {job_name}")
    client = boto3.client("batch")
    training_details = TRAINING_TYPE_MAP.get(training_type, None)
    if not training_details:
        raise Exception(f"Job Definition for training of type {training_type} is not valid.")
    response = client.submit_job(
        jobDefinition=training_details.job_definition,
        jobName=job_name,
        jobQueue=training_location.queue,
        containerOverrides={
            "command": training_details.command,
            "environment": [
                {
                    "name": "S3_PATH_TO_CONFIG",
                    "value": location,
                },
            ],
        },
    )
    print(response)
