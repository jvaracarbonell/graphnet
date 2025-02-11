"""Suggested Model subclass that enables simple user syntax."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Type

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data
import pandas as pd
from pytorch_lightning.loggers import Logger as LightningLogger

from graphnet.training.callbacks import ProgressBar
from graphnet.models.model import Model
from graphnet.models.task import StandardLearnedTask


class EasySyntax(Model):
    """A suggested Model class that comes with simple user syntax.

    This class delivers simple user syntax for training and prediction, while
    imposing minimal constraints on structure.
    """

    def __init__(
        self,
        *,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]

        # Member variable(s)
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
       

        self.validate_tasks()

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        raise NotImplementedError

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        raise NotImplementedError

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
        """
        raise NotImplementedError

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        raise NotImplementedError

    @staticmethod
    def _construct_trainer(
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> Trainer:
        if gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = 1

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            gradient_clip_val=gradient_clip_val,
            strategy=distribution_strategy,
            **trainer_kwargs,
        )

        return trainer

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        *,
        max_epochs: int = 10,
        early_stopping_patience: int = 5,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        ckpt_path: Optional[str] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> None:
        """Fit `StandardModel` using `pytorch_lightning.Trainer`."""
        # Checks
        if callbacks is None:
            # We create the bare-minimum callbacks for you.
            callbacks = self._create_default_callbacks(
                val_dataloader=val_dataloader,
                early_stopping_patience=early_stopping_patience,
            )
            self.debug("No Callbacks specified. Default callbacks added.")
        else:
            # You are on your own!
            self.debug("Initializing training with user-provided callbacks.")
            pass
        self._print_callbacks(callbacks)
        has_early_stopping = self._contains_callback(callbacks, EarlyStopping)
        has_model_checkpoint = self._contains_callback(
            callbacks, ModelCheckpoint
        )

        if (has_early_stopping) & (has_model_checkpoint is False):
            self.warning(
                "No ModelCheckpoint found in callbacks. Best-fit model will"
                " not automatically be loaded after training!"
                ""
            )

        self.train(mode=True)
        trainer = self._construct_trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )

        try:
            trainer.fit(
                self, train_dataloader, val_dataloader, ckpt_path=ckpt_path
            )
        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exiting gracefully.")
            pass

        # Load weights from best-fit model after training if possible
        if has_early_stopping & has_model_checkpoint:
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
            self.load_state_dict(
                torch.load(checkpoint_callback.best_model_path)["state_dict"]
            )
            self.info("Best-fit weights from EarlyStopping loaded.")

    def _print_callbacks(self, callbacks: List[Callback]) -> None:
        callback_names = []
        for cbck in callbacks:
            callback_names.append(cbck.__class__.__name__)
        self.info(
            f"Training initiated with callbacks: {', '.join(callback_names)}"
        )

    def _contains_callback(
        self, callbacks: List[Callback], callback: Callback
    ) -> bool:
        """Check if `callback` is in `callbacks`."""
        for cbck in callbacks:
            if isinstance(cbck, callback):
                return True
        return False

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def prediction_labels(self) -> List[str]:
        """Return prediction labels."""
        return [
            label for task in self._tasks for label in task._prediction_labels
        ]

    def configure_optimizers(self):
        """Configure the model's optimizers for both the flow (float64) and the rest (float32)."""
        
        # Find the task with the flow
        task_with_flow = None
        if hasattr(self, "_mixed_precision") and self._mixed_precision:
            for task in self._tasks:
                if hasattr(task, '_flow'):
                    task_with_flow = task
                    break
            if task_with_flow is None:
                raise ValueError("Mixed precision training requested, but no task with a flow was found.")

        self._multiple_optimizers = False
        if task_with_flow is not None:
            #This is needed if we want to use mixed precision training with float64 (flow parameters) and float32 (rest of the model)
            self.info("Training with multiple optimizers, disabling automatic_optimization")
            optimizers = []
            schedulers = []
            self.automatic_optimization = False
            self._multiple_optimizers = True
            # Optimizer for the flow (float64 parameters)
            optimizer_flow = self._optimizer_class(task_with_flow._flow.parameters(), **self._optimizer_kwargs)
            optimizers.append(optimizer_flow)

            # Get the flow parameters as a set of unique memory addresses
            flow_params = set(p for p in task_with_flow._flow.parameters())

            # Filter out flow parameters from the full list of model parameters
            rest_params = set(p for p in self.parameters()) - flow_params

            # Optimizer for the rest of the model (excluding flow parameters)
            optimizer = self._optimizer_class(list(rest_params), **self._optimizer_kwargs)
            optimizers.append(optimizer)

            # Optionally configure the learning rate scheduler for both optimizers
            if self._scheduler_class is not None:
                scheduler_flow = self._scheduler_class(optimizer_flow, **self._scheduler_kwargs)
                scheduler_rest = self._scheduler_class(optimizer, **self._scheduler_kwargs)

                # Return both optimizers and their associated schedulers in the proper format
                return [
                    {"optimizer": optimizer_flow, "lr_scheduler": scheduler_flow, "interval": "step", "monitor": "train_loss"},
                    {"optimizer": optimizer, "lr_scheduler": scheduler_rest, "interval": "step", "monitor": "train_loss"}
                ]
            return optimizers  # Return the optimizers if no schedulers are set

        else:
            optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
            )
            config = {
                "optimizer": optimizer,
            }
            if self._scheduler_class is not None:
                scheduler = self._scheduler_class(
                    optimizer, **self._scheduler_kwargs
                )
                config.update(
                    {
                        "lr_scheduler": {
                            "scheduler": scheduler,
                            **self._scheduler_config,
                        },
                    }
                )
            return config
            

    def training_step(self, train_batch: Union[Data, List[Data]], batch_idx: int) -> Tensor:
        """Perform training step, separating flow optimizer and default optimizer."""
        
        if self._multiple_optimizers:
            """ Using mixing precision float64/float32 for flow training"""
            if isinstance(train_batch, Data):
                train_batch = [train_batch]

            # Compute loss for the flow part (in float64)
            loss = self.shared_step(train_batch, batch_idx)

            # Get both optimizers
            optimizer_flow, optimizer = self.optimizers()

            # Manually perform the backward pass without relying on PyTorch Lightning's precision plugin
            # Ensure loss for the flow optimizer remains float64
            loss_float64 = loss.to(dtype=torch.float64)
            
            # Perform backward pass for the float64 part (normalizing flow)
            loss_float64.backward(retain_graph=True)

            # Convert the loss to float32 for the rest of the model
            loss_default_dtype = loss_float64.to(dtype=torch.float32)

            # Perform backward pass for the rest of the model (float32 parameters)
            loss_default_dtype.backward()

            # Now perform the optimizer steps
            optimizer_flow.step()
            optimizer_flow.zero_grad()

            optimizer.step()
            optimizer.zero_grad()

            # Log the training loss and learning rate
            self.log("train_loss", loss_default_dtype, batch_size=self._get_batch_size(train_batch), prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

            current_lr = optimizer.param_groups[0]["lr"]
            current_lr_flow = optimizer_flow.param_groups[0]["lr"]
            assert (current_lr == current_lr_flow), f"Learning rate of the two optimizers is different!!! Flows = {current_lr_flow}, normal = {current_lr}"
            self.log("lr", current_lr, prog_bar=True, on_step=True)

            return loss_default_dtype

        else:

            """Perform training step."""
            if isinstance(train_batch, Data):
                train_batch = [train_batch]
            loss = self.shared_step(train_batch, batch_idx)
            if torch.isnan(loss).any() or (loss == 0).all():
                # If NaNs are found, zero out the gradients and skip this step
                self.trainer.optimizers[0].zero_grad()
                print(f"NaN encountered at batch {batch_idx}, skipping backpropagation")
                # Optionally log NaNs occurrence
                self.log("train_loss_nan", 1, prog_bar=True, on_step=True, on_epoch=True)
                return torch.tensor(0.0, requires_grad=True)  # Return a zero tensor to avoid further issues
            else:
                # Normal training step if no NaNs are found
                self.log(
                    "train_loss",
                    loss,
                    batch_size=self._get_batch_size(train_batch),
                    prog_bar=True,
                    on_epoch=True,
                    on_step=True,
                    sync_dist=True,
                )
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log("lr", current_lr, prog_bar=True, on_step=True)
                return loss




    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform validation step."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        loss = self.shared_step(val_batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        self.train(mode=False)

        callbacks = self._create_default_callbacks(
            val_dataloader=None,
        )

        inference_trainer = self._construct_trainer(
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            callbacks=callbacks,
            **trainer_kwargs,
        )

        predictions_list = inference_trainer.predict(self, dataloader)
        assert len(predictions_list), "Got no predictions"

        nb_outputs = len(predictions_list[0])
        predictions: List[Tensor] = [
            torch.cat([preds[ix] for preds in predictions_list], dim=0)
            for ix in range(nb_outputs)
        ]
        return predictions

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        additional_attributes: Optional[List[str]] = None,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame.
        """
        if prediction_columns is None:
            prediction_columns = self.prediction_labels

        if additional_attributes is None:
            additional_attributes = []
        assert isinstance(additional_attributes, list)

        if (
            not isinstance(dataloader.sampler, SequentialSampler)
            and additional_attributes
        ):
            print(dataloader.sampler)
            raise UserWarning(
                "DataLoader has a `sampler` that is not `SequentialSampler`, "
                "indicating that shuffling is enabled. Using "
                "`predict_as_dataframe` with `additional_attributes` assumes "
                "that the sequence of batches in `dataloader` are "
                "deterministic. Either call this method a `dataloader` which "
                "doesn't resample batches; or do not request "
                "`additional_attributes`."
            )
        self.info(f"Column names for predictions are: \n {prediction_columns}")
        predictions_torch = self.predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )
        predictions = (
            torch.cat(predictions_torch, dim=1).detach().cpu().numpy()
        )
        assert len(prediction_columns) == predictions.shape[1], (
            f"Number of provided column names ({len(prediction_columns)}) and {prediction_columns} "
            f"number of output columns ({predictions.shape[1]}) don't match {predictions}."
        )

        # Check if predictions are on event- or pulse-level
        pulse_level_predictions = len(predictions) > len(dataloader.dataset)

        # Get additional attributes
        attributes: Dict[str, List[np.ndarray]] = OrderedDict(
            [(attr, []) for attr in additional_attributes]
        )
        for batch in dataloader:
            for attr in attributes:
                attribute = batch[attr]
                if isinstance(attribute, torch.Tensor):
                    attribute = attribute.detach().cpu().numpy()

                # Check if node level predictions
                # If true, additional attributes are repeated
                # to make dimensions fit
                if pulse_level_predictions:
                    if len(attribute) < np.sum(
                        batch.n_pulses.detach().cpu().numpy()
                    ):
                        attribute = np.repeat(
                            attribute, batch.n_pulses.detach().cpu().numpy()
                        )
                attributes[attr].extend(attribute)

        # Confirm that attributes match length of predictions
        skip_attributes = []
        for attr in attributes.keys():
            try:
                assert len(attributes[attr]) == len(predictions)
            except AssertionError:
                self.warning_once(
                    "Could not automatically adjust length"
                    f" of additional attribute '{attr}' to match length of"
                    f" predictions.This error can be caused by heavy"
                    " disagreement between number of examples in the"
                    " dataset vs. actual events in the dataloader, e.g. "
                    " heavy filtering of events in `collate_fn` passed to"
                    " `dataloader`. This can also be caused by requesting"
                    " pulse-level attributes for `Task`s that produce"
                    " event-level predictions. Attribute skipped."
                )
                skip_attributes.append(attr)

        # Remove bad attributes
        for attr in skip_attributes:
            attributes.pop(attr)
            additional_attributes.remove(attr)

        data = np.concatenate(
            [predictions]
            + [
                np.asarray(values)[:, np.newaxis]
                for values in attributes.values()
            ],
            axis=1,
        )

        results = pd.DataFrame(
            data, columns=prediction_columns + additional_attributes
        )
        return results

    def _create_default_callbacks(
        self,
        val_dataloader: DataLoader,
        early_stopping_patience: Optional[int] = None,
    ) -> List:
        """Create default callbacks.

        Used in cases where no callbacks are specified by the user in .fit
        """
        callbacks = [ProgressBar()]
        if val_dataloader is not None:
            assert early_stopping_patience is not None
            # Add Early Stopping
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                )
            )
            # Add Model Check Point
            callbacks.append(
                ModelCheckpoint(
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    filename=f"{self.backbone.__class__.__name__}"
                    + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                )
            )
            self.info(
                "EarlyStopping has been added"
                f" with a patience of {early_stopping_patience}."
            )
        return callbacks

    def _add_early_stopping(
        self, val_dataloader: DataLoader, callbacks: List
    ) -> List:
        if val_dataloader is None:
            return callbacks
        has_early_stopping = False
        assert isinstance(callbacks, list)
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                has_early_stopping = True

        if not has_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                )
            )
            self.warning_once(
                "Got validation dataloader but no EarlyStopping callback. An "
                "EarlyStopping callback has been added automatically with "
                "patience=5 and monitor = 'val_loss'."
            )
        return callbacks
