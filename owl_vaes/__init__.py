from .configs import Config
from .models import get_model_cls
from .utils import versatile_load

def from_pretrained(cfg_path, ckpt_path):
    cfg = Config.from_yaml(cfg_path).model
    model_cls = get_model_cls(cfg.model_id)
    model = model_cls(cfg)
    
    ckpt = versatile_load(ckpt_path)
    try:
        model.load_state_dict(ckpt)
    except:
        # Perhaps its encoder only or decoder only
        try:
            model.encoder.load_state_dict(ckpt)
            print("Warning: Checkpoint is encoder only")
        except:
            try:
                model.decoder.load_state_dict(ckpt)
                print("Warning: Checkpoint is decoder only")
            except:
                raise ValueError(f"Could not load state dict for {ckpt_path}")
            
    return model