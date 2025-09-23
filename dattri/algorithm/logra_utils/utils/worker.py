from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, Subset


def get_worker_batch_range(
    total_batches: int,
    chunk_size: int,
    worker: str,
) -> Tuple[int, int]:
    """Get chunk-aligned batch range for a worker.

    Args:
        total_batches: Total number of batches in the dataset.
        chunk_size: Size of each chunk for processing.
        worker: Worker specification in format "worker_id/total_workers".

    Returns:
        Tuple[int, int]: A tuple of (start_batch, end_batch) for this worker.

    Raises:
        ValueError: If worker specification is invalid or worker_id is out of range.
    """
    try:
        worker_id, total_workers = map(int, worker.split("/"))

        if worker_id < 0 or total_workers <= 0 or worker_id >= total_workers:
            msg = "Invalid worker specification"
            raise ValueError(msg)

    except ValueError as e:
        msg = f"Invalid worker specification '{worker}': {e}"
        raise ValueError(msg)

    # Calculate total chunks
    total_chunks = (total_batches + chunk_size - 1) // chunk_size

    # Distribute chunks among workers
    chunks_per_worker = total_chunks // total_workers
    remaining_chunks = total_chunks % total_workers

    # Calculate chunk range for this worker
    start_chunk = worker_id * chunks_per_worker + min(worker_id, remaining_chunks)
    end_chunk = (
        start_chunk + chunks_per_worker + (1 if worker_id < remaining_chunks else 0)
    )

    # Convert to batch range
    start_batch = start_chunk * chunk_size
    end_batch = min(end_chunk * chunk_size, total_batches)

    return start_batch, end_batch


def create_worker_dataloader(
    dataloader: "DataLoader",
    start_batch: int,
    end_batch: int,
) -> "DataLoader":
    """Create an efficient subset dataloader for this worker's batch range.

    Args:
        dataloader: Original DataLoader to create a subset from.
        start_batch: Starting batch index for this worker.
        end_batch: Ending batch index for this worker.

    Returns:
        DataLoader: A new DataLoader containing only the subset of data for this worker.
    """
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size

    # Calculate sample indices for this batch range
    start_idx = start_batch * batch_size
    end_idx = min(end_batch * batch_size, len(dataset))

    # Create subset indices - this is just a list of integers!
    indices = list(range(start_idx, end_idx))
    subset = Subset(dataset, indices)

    # Create new DataLoader with same settings but using the subset
    return type(dataloader)(
        subset,
        batch_size=batch_size,
        shuffle=False,  # Important: don't shuffle for worker consistency
        num_workers=0,  # Avoid nested multiprocessing
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        # Copy other relevant attributes as needed
    )
