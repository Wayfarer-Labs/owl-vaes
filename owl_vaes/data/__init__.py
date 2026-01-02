def get_loader(data_id: str, batch_size: int, **data_kwargs):
    if data_id == "mnist":
        from . import mnist
        return mnist.get_loader(batch_size)
    if data_id == "video_dir_loader":
        from .video_dir_loader import get_loader
        return get_loader(batch_size, **data_kwargs)
    if data_id == "video_dir_audio_loader":
        from .video_dir_audio_loader import get_loader
        return get_loader(batch_size, **data_kwargs)