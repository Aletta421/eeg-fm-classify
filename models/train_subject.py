"""Train REVE with one loss and one prediction per subject."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import (
    ExperimentConfig,
    get_exp_adhd,
    get_exp_depression,
    get_exp_single_dataset,
)
from models.subject_dataset import create_subject_dataloaders
from models.subject_model import SubjectAggregationModel
from models.subject_trainer import SubjectTrainer

logger = logging.getLogger(__name__)


def build_reve_encoder(config, device):
    """Load pretrained REVE and fail rather than silently using random weights."""
    from braindecode.models import REVE

    if config.hf_endpoint:
        os.environ["HF_ENDPOINT"] = config.hf_endpoint
    model = REVE.from_pretrained(
        config.model_id,
        force_download=config.force_download,
        local_files_only=config.local_files_only,
        token=config.hf_token,
        n_outputs=config.n_outputs,
        n_chans=None,
        n_times=config.n_times,
        input_window_seconds=config.input_window_seconds,
        sfreq=config.sfreq,
        attention_pooling=config.use_attention_pooling,
    )
    if not hasattr(model, "final_layer"):
        raise AttributeError("REVE model does not expose final_layer")
    model.final_layer = nn.Identity()
    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["depression", "adhd"])
    parser.add_argument("--dataset")
    parser.add_argument("--diagnosis", choices=["depression", "adhd"])
    parser.add_argument("--datasets", help="comma-separated dataset override")
    parser.add_argument(
        "--aggregation",
        choices=["mean", "transform_mean"],
        default="mean",
    )
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--subject-batch-size", type=int, default=4)
    parser.add_argument("--train-segments", type=int, default=8)
    parser.add_argument("--eval-segments", type=int, default=-1)
    parser.add_argument("--encoder-chunk-size", type=int, default=32)
    parser.add_argument("--transform-hidden-dim", type=int, default=256)
    parser.add_argument("--transform-dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-channels", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-mirror")
    return parser.parse_args()


def build_experiment(args) -> ExperimentConfig:
    if args.experiment == "depression":
        config = get_exp_depression()
    elif args.experiment == "adhd":
        config = get_exp_adhd()
    elif args.dataset:
        config = get_exp_single_dataset(args.dataset, args.diagnosis)
    else:
        config = ExperimentConfig(name="exp_subject_all")
    if args.datasets:
        config.datasets = [
            name.strip() for name in args.datasets.split(",") if name.strip()
        ]
    config.name = args.name or f"{config.name}_{args.aggregation}"
    config.device = args.device
    config.data.num_workers = args.num_workers
    config.data.max_channels = args.max_channels
    config.model.local_files_only = args.local_files_only
    config.model.hf_endpoint = args.hf_mirror
    config.checkpoint_dir = (
        config.output_root / "checkpoints" / config.name
    )
    return config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    config = build_experiment(args)
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cpu":
        logger.warning(
            "Running with CPU. Use this only for smoke tests; REVE training needs GPU."
        )

    loaders = create_subject_dataloaders(
        labels_csv=config.data.labels_csv,
        include_diagnosis=config.data.include_diagnosis or None,
        include_datasets=config.datasets or None,
        exclude_datasets=config.data.exclude_datasets or None,
        max_channels=config.data.max_channels,
        val_fraction=args.val_fraction,
        train_segments_per_subject=args.train_segments,
        eval_segments_per_subject=args.eval_segments,
        batch_size=args.subject_batch_size,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
        project_root=config.data.project_root,
    )
    for split in ("train", "val", "test"):
        logger.info(
            "%s subjects: %d",
            split,
            len(loaders[f"{split}_dataset"]),
        )
    if not all(len(loaders[f"{split}_dataset"]) for split in ("train", "val", "test")):
        raise ValueError("train, val, and test must each contain at least one subject")

    encoder = build_reve_encoder(config.model, device)
    model = SubjectAggregationModel(
        encoder=encoder,
        embedding_dim=512,
        n_classes=2,
        aggregation=args.aggregation,
        transform_hidden_dim=args.transform_hidden_dim,
        transform_dropout=args.transform_dropout,
        freeze_encoder=True,
        encoder_chunk_size=args.encoder_chunk_size,
    )
    trainer = SubjectTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=loaders["train_dataset"].class_weights,
        patience=args.patience,
    )
    run_config = {
        "experiment": config.name,
        "datasets": config.datasets,
        "diagnosis": config.data.include_diagnosis,
        "aggregation": args.aggregation,
        "val_fraction": args.val_fraction,
        "train_segments_per_subject": args.train_segments,
        "eval_segments_per_subject": args.eval_segments,
        "subject_batch_size": args.subject_batch_size,
        "encoder_chunk_size": args.encoder_chunk_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_id": config.model.model_id,
    }
    result = trainer.fit(
        loaders["train"],
        loaders["val"],
        loaders["test"],
        epochs=args.epochs,
        checkpoint_dir=config.checkpoint_dir,
        config=run_config,
    )
    logger.info("Subject-level result:\n%s", json.dumps(result["final"], indent=2))


if __name__ == "__main__":
    main()
