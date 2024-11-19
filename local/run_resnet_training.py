import torch

from local.training import execute_training
from local.training_struct import TrainingLocation
from models.common.base_model_config import BaseDataConfig
from models.resnet.resnet_configs import ResnetModelConfig
from models.resnet.resnet_entrypoint import train_resnet


from models.resnet.resnet_full_config import ResnetFullConfig


def do_work():
    training_location = TrainingLocation.LOCAL.value
    output_path = training_location.value.output_path

    experiment_name = "test_run"
    tensor_version = 0
    dataset_name ='test_run'
    input_path = training_location.input_path +f'/version_{tensor_version}'

    config = ResnetFullConfig(
        model_config=ResnetModelConfig(
            device=torch.device("cpu"),
            lr_scheduler_mode="min",# has to be 'min' or 'max',
            lr_scheduler_factor=0.1,
            lr_scheduler_patience=2,
            reduction='mean',
            head_layers=[1000,10,2],
            activation='relu',
            num_epochs=2,
            learning_rate=0.0001,
            checkpoint_path=None,
            # checkpoint_path='s3://playai-cv-video-filter/training/mvit_pubg_other_full_logits/lightning_logs/version_3/checkpoints/epoch=34-step=4095.ckpt',
            loss='cross_entropy',#,'cross_entropy',#'focal_loss',#"cross_entropy",
            step_size=10,
            gamma=0.1,
            focal_loss_alpha=0.1,
            focal_loss_gamma=0.1,
            loss_weight=[1.0,1.0,1.0,1.0,1.0,1.0],
        ),
        data_config=BaseDataConfig(
            experiment_descriptor=experiment_name,
            input_path=input_path,
            output_path=output_path,
            # download_data=download_data,
            batch_size=4,
            test_batch_size=1,
            time_window=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            num_workers=4,
            prefetch_factor=1,
            num_classes=2,
            # s3_tensor_path=f"s3://playai-cv-video-filter/data/yt_pubg_cs_data/version_{tensor_version}/tensors",
            background_class=0,
            split_file_name='dummy_split',
        ),
    )

    print(training_location)
    execute_training(config=config, execution=train_resnet(), location=training_location)


if __name__ == "__main__":
    do_work()
