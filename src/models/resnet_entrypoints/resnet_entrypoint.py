import json
import os
import sys

import torch.backends.mps


from pytorch_model_framework.model.model_driver import CommandLineArgs, parse_cli_args, ModelDriver


def get_data_module(config: MvitDataConfig, stage=None):
    data_module = BaseDataModule(
        data_config=config,
        transforms_x=setup_transforms_from_list(config.x_transforms_names),
        transforms_y=setup_transforms_from_list(config.y_transforms_names),
        common_transforms=setup_common_transforms_from_list(
            transforms_names=config.common_transforms_name,
            arguments_list=[
                dict(
                    num_times=config.num_tokens_to_replicate,
                    static_video_probability=config.static_video_probability,
                )
            ]
        )
    )
    data_module.setup(stage=stage)
    return data_module


def build_config(model_config_path: str) -> FullMViTConfig:
    if os.path.isfile(model_config_path):
        with open(model_config_path, "r") as f:
            config_json = json.loads(f.read())
            return FullMViTConfig.from_dict(input_dict=config_json)
    else:
        raise RuntimeError(f"No file exists: {model_config_path}")


def freeze_custom(model, train_config):
    layers = list(model.children())
    layers = list(layers[0].children())
    # layers = list(self.backbone.children())
    for layer in layers[:2]:
        for params in layer.parameters():
            params.requires_grad = False
    tformer_blocks = list(layers[2].children())

    for num_block, block in enumerate(tformer_blocks):
        if num_block >= train_config.freeze_num_tformer_blocks:
            break
        for layer in block.children():
            for params in layer.parameters():
                params.requires_grad = False
    return model


def get_loss_function(train_config: MvitModelConfig):
    loss = LOSSES[train_config.loss.lower()].value
    if train_config.loss.lower() == 'bce_loss':
        return loss(reduction=train_config.reduction)
    if train_config.loss.lower() == 'ce_still_loss':
        return loss(num_classes=train_config.num_classes,reduction=train_config.reduction, weight=torch.tensor(train_config.loss_weight))
    return loss(reduction=train_config.reduction, weight=torch.tensor(train_config.loss_weight))


def get_head_module(train_config: MvitModelConfig):
    head = LinearHead(linear_layers= train_config.head_layers, activation=train_config.activation, last_layer_activation= train_config.last_activation)
    # head = head_module(train_config=train_config)
    return head


def train_mvit(cli_args: CommandLineArgs):
    config = build_config(model_config_path=cli_args.model_config_path)
    data_module = get_data_module(config=config.data_config)

    model = MViTModel(
        model_config=config.model_config,
        head=get_head_module(train_config=config.model_config),
        loss=get_loss_function(config.model_config),
    )

    # model = freeze_custom(model, config.model_config)
    ModelDriver(
        data_config=config.data_config, model_config=config.model_config, data_module=data_module, model=model
    ).run_training()


if __name__ == "__main__":
    args = parse_cli_args(sys.argv)
    # args = CommandLineArgs(os.path.join(ROOT_DIR,"data/training/input_config.json"))
    train_mvit(cli_args=args)
