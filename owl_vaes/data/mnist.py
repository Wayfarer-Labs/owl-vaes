import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader

def get_loader(batch_size):

    """Get MNIST data loader

    Args:
        batch_size: Batch size per GPU
        world_size: Number of GPUs being used
        global_rank: Rank of current GPU

    Returns:
        DataLoader for MNIST dataset
    """

    world_size = 1
    global_rank = 0
    if dist.is_initialized():
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

    dataset = load_dataset("mnist")
    train_dataset = dataset["train"]

    def collate_fn(examples):
        imgs = [example['image'] for example in examples]
        imgs = [img.resize((64, 64), Image.BILINEAR) for img in imgs]
        tensors = [TF.to_tensor(img) * 2 - 1 for img in imgs]
        batch = torch.stack(tensors)
        batch = batch.repeat(1, 3, 1, 1)

        return batch

    if world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        generator=torch.Generator('cuda')
    )

    return train_loader
