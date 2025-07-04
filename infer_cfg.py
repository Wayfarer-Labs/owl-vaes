import torch
import yaml
import sys
from pathlib import Path

def infer_config_from_checkpoint(ckpt_path):
    """Infer model config parameters from a checkpoint file"""
    
    # Load checkpoint
    d = torch.load(ckpt_path, map_location='cpu')
    
    # Handle EMA and DDP prefixes
    prefix = ""
    if 'ema' in d:
        d = d['ema']
        prefix += "ema_model."
    if any('.module.' in k for k in d.keys()):
        prefix += "module."
    
    # Strip prefixes from keys
    state_dict = {k[len(prefix):] : v for k,v in d.items() if k.startswith(prefix)}

    # Start building config
    config = {
        'model': {
            'model_id': 'dcae',
            'sample_size': [360, 640],
            'channels': 3,
        }
    }

    # Infer latent dimensions from encoder output conv
    conv_out = state_dict.get('encoder.conv_out.weight')
    if conv_out is not None:
        config['model']['latent_channels'] = conv_out.shape[0]

    # Get initial channel count from first conv
    conv_in = state_dict.get('encoder.conv_in.weight')
    if conv_in is not None:
        config['model']['ch_0'] = conv_in.shape[0]

    # Find maximum channel dimension in state dict
    ch_max = 0
    for k, v in state_dict.items():
        if 'weight' in k and len(v.shape) > 1:
            ch_max = max(ch_max, v.shape[0], v.shape[1])
    config['model']['ch_max'] = ch_max

    # Count encoder/decoder blocks by looking at module structure
    enc_blocks = []
    dec_blocks = []
    
    enc_stage = 0
    dec_stage = 0
    while True:
        enc_key = f'encoder.blocks.{enc_stage}.blocks.0.conv1.weight'
        if enc_key not in state_dict:
            break
        block_count = 0
        while f'encoder.blocks.{enc_stage}.blocks.{block_count}.conv1.weight' in state_dict:
            block_count += 1
        enc_blocks.append(block_count)
        enc_stage += 1

    while True:
        dec_key = f'decoder.blocks.{dec_stage}.blocks.0.conv1.weight'
        if dec_key not in state_dict:
            break
        block_count = 0
        while f'decoder.blocks.{dec_stage}.blocks.{block_count}.conv1.weight' in state_dict:
            block_count += 1
        dec_blocks.append(block_count)
        dec_stage += 1

    config['model']['encoder_blocks_per_stage'] = enc_blocks
    config['model']['decoder_blocks_per_stage'] = dec_blocks

    # Infer latent size from encoder output shape
    # For now hardcoding to 4 since we'd need a forward pass to determine this
    config['model']['latent_size'] = 4

    return config

def save_config(config, save_path):
    """Save config dict as YAML"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer_cfg.py <checkpoint_path> <save_path>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    save_path = sys.argv[2]

    config = infer_config_from_checkpoint(ckpt_path)
    save_config(config, save_path)
    print(f"Config saved to {save_path}")
