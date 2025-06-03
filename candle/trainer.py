import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from safetensors.torch import save_file

class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_dataset,
        eval_dataset=None,
        batch_size=32,
        num_epochs=10,
        evaluation_strategy="epoch",  # or int
        metric_fn=None,
        device=None,
        save_strategy="epoch",  # or int
        save_folder="training"
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluation_strategy = evaluation_strategy
        self.metric_fn = metric_fn if metric_fn is not None else self.loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_strategy = save_strategy
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.model.to(self.device)

    def _save_model(self, subfolder):
        save_path = os.path.join(self.save_folder, subfolder)
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, "model.safetensors")
        save_file(self.model.state_dict(), model_file)

    def _evaluate(self):
        self.model.eval()
        eval_loader = DataLoader(self.eval_dataset, batch_size=self.batch_size)
        total_loss = 0.0
        total_metric = 0.0
        count = 0
        with torch.no_grad():
            for batch in eval_loader:
                label_args = batch["label"]
                label_args = {k: v.to(self.device) for k, v in label_args.items()}
                preds = self.model(**label_args)
                if "target" in batch:
                    target_args = batch["target"]
                    target_args = {k: v.to(self.device) for k, v in target_args.items()}
                    loss = self.loss_fn(preds, **target_args)
                    metric = self.metric_fn(preds, **target_args)
                    batch_size = next(iter(target_args.values())).size(0)
                else:
                    loss = self.loss_fn(preds)
                    metric = self.metric_fn(preds)
                    batch_size = next(iter(label_args.values())).size(0)
                total_loss += loss.item() * batch_size
                total_metric += metric.item() * batch_size
                count += batch_size
        avg_loss = total_loss / count
        avg_metric = total_metric / count
        return avg_loss, avg_metric

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            num_batches = len(train_loader)
            pbar = tqdm(enumerate(train_loader, 1), total=num_batches, desc=f"Epoch {epoch}/{self.num_epochs}")
            for batch_idx, batch in pbar:
                label_args = batch["label"]
                label_args = {k: v.to(self.device) for k, v in label_args.items()}
                preds = self.model(**label_args)
                if "target" in batch:
                    target_args = batch["target"]
                    target_args = {k: v.to(self.device) for k, v in target_args.items()}
                    loss = self.loss_fn(preds, **target_args)
                    batch_size = next(iter(target_args.values())).size(0)
                else:
                    loss = self.loss_fn(preds)
                    batch_size = next(iter(label_args.values())).size(0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_size

                # Evaluation logic
                eval_loss = eval_metric = None
                do_eval = False
                if self.eval_dataset is not None:
                    if self.evaluation_strategy == "epoch":
                        do_eval = False  # only at end of epoch
                    elif isinstance(self.evaluation_strategy, int):
                        if batch_idx % self.evaluation_strategy == 0:
                            do_eval = True
                if do_eval:
                    eval_loss, eval_metric = self._evaluate()

                # Model saving logic
                do_save = False
                if self.save_strategy == "epoch":
                    do_save = False  # only at end of epoch
                elif isinstance(self.save_strategy, int):
                    if batch_idx % self.save_strategy == 0:
                        do_save = True
                if do_save:
                    self._save_model(f"batch_{batch_idx}")

                # Progress bar update
                postfix = {
                    "train_loss": loss.item()
                }
                if eval_loss is not None:
                    postfix["eval_loss"] = eval_loss
                if eval_metric is not None:
                    postfix["eval_metric"] = eval_metric
                pbar.set_postfix(postfix)

            # End-of-epoch evaluation
            avg_epoch_loss = epoch_loss / len(self.train_dataset)
            eval_loss = eval_metric = None
            if self.eval_dataset is not None and self.evaluation_strategy == "epoch":
                eval_loss, eval_metric = self._evaluate()
            tqdm.write(
                f"Epoch {epoch} done. Train loss: {avg_epoch_loss:.4f}"
                + (f", Eval loss: {eval_loss:.4f}, Eval metric: {eval_metric:.4f}" if eval_loss is not None else "")
            )

            # Save model at end of epoch
            if self.save_strategy == "epoch":
                self._save_model(f"epoch_{epoch}")

        # Save final model at end of training
        self._save_model("final")
