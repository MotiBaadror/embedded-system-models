from collections import defaultdict
from datetime import datetime
from typing import Optional

import datasets
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW


class GLUETransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue",
            self.hparams.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
            trust_remote_code=True,
        )
        self.outputs = defaultdict(list)
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')



    def forward(self, **inputs):
        return self.model(**inputs)

    def get_cls(self, pred):
        _, pred_labels = torch.max(pred, dim=-1)
        return pred_labels

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        pred_labels = self.get_cls(pred=outputs[1])
        self.train_accuracy.update(pred_labels, batch['labels'])
        loss = outputs[0]
        self.log('training loss', value=loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        pred_labels = self.get_cls(pred=logits)
        self.val_accuracy.update(pred_labels, batch['labels'])
        self.log('val loss', value=val_loss,on_step=True,on_epoch=True)

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        self.outputs[dataloader_idx].append({"loss": val_loss, "preds": preds, "labels": labels})

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_accuracy.compute(), on_epoch=True, prog_bar = True, logger = True)
        self.train_accuracy.reset()
    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_accuracy.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.val_accuracy.reset()
        if self.hparams.task_name == "mnli":
            for i, outputs in self.outputs.items():
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in outputs]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        preds = torch.cat([x["preds"] for x in flat_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in flat_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)."""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]