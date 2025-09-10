def get_loader(data_id: str, batch_size: int, **data_kwargs):
    if data_id == "mnist":
        from . import mnist
        return mnist.get_loader(batch_size)
    if data_id == "local_imagenet_256":
        from . import local_imagenet_256
        return local_imagenet_256.get_loader(batch_size)
    if data_id == "s3_imagenet":
        from . import imagenet
        return imagenet.get_loader(batch_size)
    if data_id == "local_cod":
        from . import local_cod
        return local_cod.get_loader(batch_size)
    if data_id == "audio_loader":
        from .audio_loader import get_audio_loader
        assert data_kwargs['filepath'] is not None, "No filepath provided for the dataset."
        return get_audio_loader(batch_size, data_kwargs['filepath'])
    if data_id == "local_cod_audio":
        from .local_cod_audio import get_loader
        return get_loader(batch_size, **data_kwargs)
    if data_id == "s3_cod_audio":
        from .s3_audio import get_loader
        return get_loader(batch_size, **data_kwargs)
    if data_id == "s3_cod_features":
        from .s3_cod_features import get_loader
        #from .s3_cod_features_shuffle import get_loader
        return get_loader(batch_size, **data_kwargs)
    if data_id == "local_cod_features":
        from .local_cod_features import get_loader
        return get_loader(batch_size, **data_kwargs)
    if data_id == "local_wf_data":
        from .local_wf_data import get_loader
        return get_loader(batch_size, **data_kwargs)
    if data_id == "local_latent":
        from .local_diffdec_pairs import get_loader
        return get_loader(batch_size, **data_kwargs)
