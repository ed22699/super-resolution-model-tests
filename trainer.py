import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path
import sys
from dataloader import DIV2KDataLoader
from gan import GAN
import os


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a chosen model on DIV2K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path(os.getcwd() + "/DIV2K")
parser.add_argument(
    "--dataset-root",
    default=default_dataset_dir,
    help="The location of the dataset to be trained on")
parser.add_argument(
    "--log-dir",
    default=Path("logs"),
    type=Path,
    help="The path to the folder containing training logs for tensorboard")
parser.add_argument(
    "--model",
    default="GAN",
    type=str,
    help="choose between GAN, Diffusion and Transformer")
parser.add_argument(
    "--learning-rate",
    default=1e-2,
    type=float,
    help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=16,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--patience",
    default=5,
    type=int,
    help="Number of higher val losses to trigger early stopping",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--weight_decay",
    default=0,
    type=float,
    help="weight_decay used for ADAM Optimiser",
)
parser.add_argument(
    "--beta1",
    default=0.9,
    type=float,
    help="Beta1 used for ADAM optimiser",
)
parser.add_argument(
    "--beta2",
    default=0.999,
    type=float,
    help="Beta2 used for ADAM optimiser",
)
parser.add_argument(
    "--eps",
    default=1e-8,
    type=float,
    help="eps (epsillon, to prevent dividing by 0) used for ADAM optimiser",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):

    transformList = []
    transformList.append(transforms.ToTensor())
    basic_transforms = transforms.Compose(transformList)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    datasetRoot = str(args.dataset_root)
    lr_path = datasetRoot+"/DIV2K_train_LR_bicubic/X8"
    hr_path = datasetRoot+"/DIV2K_train_HR"

    train_dataset = DIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transform=basic_transforms,
        mode="train",
        batch_size=args.batch_size
    )

    lr_path = datasetRoot+"/DIV2K_valid_LR_bicubic/x8"
    hr_path = datasetRoot+"/DIV2K_valid_HR"

    val_dataset = DIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transform=basic_transforms,
        mode="val",
        batch_size=args.batch_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # TODO fill in parameters for the different models
    if args.model == "GAN":
        model = GAN()
    else:
        print("Model currently not implemented")
        return

    criterion = nn.CrossEntropyLoss()

    betas = (args.beta1, args.beta2)
    optimizer = torch.optim.AdamW(model.parameters(
    ), lr=args.learning_rate, betas=betas, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE, scheduler
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
        patience=args.patience,
    )

    summary_writer.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        scheduler,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
        patience: int = 1,
    ):
        self.model.train()
        # Early stopping conditions
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch_lr, batch_hr in self.train_loader:
                batch_lr = batch_lr.to(self.device)
                batch_hr = batch_hr.to(self.device)
                data_load_end_time = time.time()

                # Compute the forward pass of the model
                logits = self.model.forward(batch_lr)

                # TODO set up loss to be suitable
                # Compute the loss
                loss = self.criterion(logits, batch_hr)

                # Compute the backward pass
                loss.backward()

                # Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                # TODO set up accuracy to be suitable
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss,
                                     data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss,
                                       data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # Early Stopping Code
            if ((epoch + 1) % val_frequency) == 0:
                val_loss = self.validate()
                # switch back to train mode afterwards
                # self.model.train()
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "best_model.pth")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("Early stopping triggered!")
                        break

            self.scheduler.step()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "accuracy",
            {"train": accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"train": float(loss.item())},
            self.step
        )
        self.summary_writer.add_scalar(
            "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
            "time/data", step_time, self.step
        )

    def validate(self) -> float:
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch_anc, batch_comp, labels in self.val_loader:
                batch_anc = batch_anc.to(self.device)
                batch_comp = batch_comp.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch_anc, batch_comp)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
            "accuracy",
            {"test": accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"test": average_loss},
            self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {
              accuracy * 100:2.2f}")
        classAcc = compute_per_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        for c in range(len(classifications)):
            print(f"{classifications[c]} accuracy: {classAcc[c] * 100:2.2f}")

        return average_loss

    def run_overfit_batch_test(self, steps: int):
        self.model.train()

        # 1. Grab a SINGLE batch from the loader
        data_iter = iter(self.train_loader)
        batch_anc, batch_comp, labels = next(data_iter)

        # 2. Move to GPU once
        batch_anc = batch_anc.to(self.device)
        batch_comp = batch_comp.to(self.device)
        labels = labels.to(self.device)

        print(f"Overfitting on a single batch of size {len(labels)}...")

        start_time = time.time()

        for i in range(steps):
            # Forward pass
            logits = self.model.forward(batch_anc, batch_comp)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate Accuracy
            with torch.no_grad():
                preds = logits.argmax(-1)
                accuracy = compute_accuracy(labels, preds)

            # Print every 5 steps
            if (i + 1) % 5 == 0:
                print(f"Step [{i+1}/{steps}] - Loss: {loss.item()                      :.6f} - Acc: {accuracy * 100:.2f}%")

                # 3. Early exit if we solved it
                if accuracy == 1.0 and loss.item() < 0.01:
                    print("\nSUCCESS: Model successfully memorized the batch!")
                    return

        print("\nTest Finished.")
        if accuracy < 1.0:
            print(
                "WARNING: Model failed to memorize the batch. Check LR, labels, or architecture.")


def test_model_accuracy(model: nn.Module, test_loader: DataLoader, device: torch.device, criterion: nn.Module):
    results = {"preds": [], "labels": []}
    total_loss = 0
    # weights_only is passed to torch.load (Right)
    state_dict = torch.load("best_model.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch_anc, batch_comp, labels in test_loader:
            batch_anc = batch_anc.to(device)
            batch_comp = batch_comp.to(device)
            labels = labels.to(device)
            logits = model(batch_anc, batch_comp)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            results["preds"].extend(list(preds))
            results["labels"].extend(list(labels.cpu().numpy()))

    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    average_loss = total_loss / len(test_loader)

    print("-----Model Accuracy/Loss-----")
    print(f"test loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
    classAcc = compute_per_class_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    for c in range(len(classifications)):
        print(f"{classifications[c]} accuracy: {classAcc[c] * 100:2.2f}")


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def compute_per_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> [float]:
    assert len(labels) == len(preds)

    acc = np.array([0 for c in classifications], dtype=float)
    unique, counts = np.unique(labels, return_counts=True)
    denoms = dict(zip(unique, counts))

    for c in range(len(classifications)):
        acc[c] = float(((labels == preds) * (preds == c)).sum()) / denoms[c]
    return acc


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"dropout={0.5}_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"weight_decay={args.weight_decay}_"
        f"beta1={args.beta1}_"
        f"beta2={args.beta2}_"
        f"eps={args.eps}_"
        f"brightness={args.data_aug_brightness}_" +
        f"crop={args.data_aug_crop}_" +
        ("perspec_" if args.data_aug_perspect else "") +
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
