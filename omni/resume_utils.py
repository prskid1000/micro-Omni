from torch.utils.data import DataLoader


def setup_resume_data_loading(
    train_ds,
    step: int,
    batch_size: int,
    logger,
    train_dl_kwargs,
    *,
    train_size: int | None = None,
    drop_last: bool = True,
    accumulation_steps: int = 1,
):
    if step > 0:
        accumulation_steps = max(int(accumulation_steps), 1)
        micro_batches_seen = int(step) * accumulation_steps
        skip_samples = 0
        if train_size is not None and train_size > 0:
            steps_per_epoch = train_size // batch_size
            if not drop_last and (train_size % batch_size) != 0:
                steps_per_epoch += 1
            steps_per_epoch = max(int(steps_per_epoch), 1)
            start_batch_idx = micro_batches_seen % steps_per_epoch
            skip_samples = int(start_batch_idx) * int(batch_size)
        else:
            logger.info("Dataset size unknown; resume will not skip samples (starts at epoch boundary).")
        underlying_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
        underlying_ds.skip_samples = skip_samples
        logger.info(
            f"Dataset: resume at optimizer_step={step} (accum={accumulation_steps}) "
            f"-> micro_batches_seen≈{micro_batches_seen}, skip_samples={skip_samples}"
        )
        train_dl_kwargs["batch_size"] = batch_size
        return DataLoader(train_ds, **train_dl_kwargs)
    return None


def calculate_resume_position(step, steps_per_epoch):
    if step > 0:
        return step // steps_per_epoch, step % steps_per_epoch
    return 0, 0


def calculate_steps_per_epoch(train_size: int | None, batch_size: int, drop_last: bool) -> int | None:
    if train_size is None:
        return None
    batch_size = max(int(batch_size), 1)
    steps = train_size // batch_size
    if (not drop_last) and (train_size % batch_size) != 0:
        steps += 1
    return max(int(steps), 1)


def calculate_micro_batches_seen(optimizer_step: int, accumulation_steps: int) -> int:
    return int(optimizer_step) * max(int(accumulation_steps), 1)


def calculate_resume_epoch_batch_from_optimizer_step(
    optimizer_step: int,
    accumulation_steps: int,
    steps_per_epoch: int,
) -> tuple[int, int]:
    micro = calculate_micro_batches_seen(optimizer_step, accumulation_steps)
    return calculate_resume_position(micro, steps_per_epoch)


class ValidationSkipSamplesContext:
    """Context manager to temporarily reset skip_samples for validation."""

    def __init__(self, train_ds):
        self.train_ds = train_ds
        self.underlying_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
        self.original_skip_samples = None

    def __enter__(self):
        if hasattr(self.underlying_ds, "skip_samples"):
            self.original_skip_samples = self.underlying_ds.skip_samples
            self.underlying_ds.skip_samples = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.underlying_ds, "skip_samples") and self.original_skip_samples is not None:
            self.underlying_ds.skip_samples = self.original_skip_samples

__all__ = [
    "setup_resume_data_loading",
    "calculate_resume_position",
    "calculate_steps_per_epoch",
    "calculate_micro_batches_seen",
    "calculate_resume_epoch_batch_from_optimizer_step",
    "ValidationSkipSamplesContext",
]
