import os
import glob
import torch
import random
from torch.utils.data import IterableDataset, DataLoader


class LocalWaveformDataset(IterableDataset):
    def __init__(self, root_dir, window_length=44100):
        super().__init__()
        self.root_dir = root_dir
        self.window_length = window_length
        
        # Build list of all waveform tensor files
        self.wf_files = []
        pattern = os.path.join(root_dir, "**", "*_wf.pt")
        self.wf_files = glob.glob(pattern, recursive=True)
        
        if not self.wf_files:
            raise ValueError(f"No waveform files found in {root_dir}")
        
        print(f"Found {len(self.wf_files)} waveform files")

    def __iter__(self):
        while True:
            # Pick a random file
            file_path = random.choice(self.wf_files)
            
            try:
                # Load tensor with mmap for efficiency
                waveform = torch.load(file_path, map_location='cpu', mmap=True)
                
                # Get random sample
                if waveform.shape[0] >= self.window_length:
                    start_idx = random.randint(0, waveform.shape[0] - self.window_length)
                    sample = waveform[start_idx:start_idx + self.window_length]
                    yield sample
                else:
                    # If file is shorter than window, pad with zeros
                    padded = torch.zeros((self.window_length, waveform.shape[1]), dtype=waveform.dtype)
                    padded[:waveform.shape[0]] = waveform
                    yield padded
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue


def collate_fn(batch):
    # Stack waveforms into [batch, window_length, 2] tensor
    return torch.stack(batch).transpose(1,2) # [b,2,n]


def get_loader(batch_size, **data_kwargs):
    root_dir = data_kwargs['root_dir']
    window_length = data_kwargs.get('window_length', 44100)
    
    ds = LocalWaveformDataset(root_dir=root_dir, window_length=window_length)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)