from owl_vaes.data.audio_loader import get_audio_loader

from . import imagenet, local_imagenet_256, mnist


def get_loader(data_id: str, batch_size: int, filepath: str | list[str] | None = None):
    if data_id == "mnist":
        return mnist.get_loader(batch_size)
    if data_id == "local_imagenet_256":
        return local_imagenet_256.get_loader(batch_size)
    if data_id == "s3_imagenet":
        return imagenet.get_loader(batch_size)
    if data_id == "audio_loader":
        assert filepath is not None, "No filepath provided for the dataset."
        return get_audio_loader(batch_size, filepath)
