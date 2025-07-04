import torch
import torch.nn.functional as F
import pytorch_wavelets as pw

def dwt_loss_fn(x, y):
    """
    Compute DWT loss between two images using L1 distance between wavelet coefficients
    
    Args:
        x: First image tensor [B,C,H,W]
        y: Second image tensor [B,C,H,W]
        
    Returns:
        Average L1 loss between wavelet coefficients
    """
    # Create wavelet transform with 4 levels
    xfm = pw.DWTForward(J=4, wave='haar').to(x.device)
    
    # Get coefficients for both images
    # LL is in yl, (LH,HL,HH) are in yh
    yl_x, yh_x = xfm(x.float())
    yl_y, yh_y = xfm(y.float())
    
    # L1 loss on low frequency component (LL)
    ll_loss = torch.abs(yl_x - yl_y).mean()
    
    # L1 loss on high frequency components (LH,HL,HH) at each level
    hf_loss = 0
    for level_x, level_y in zip(yh_x, yh_y):
        hf_loss += torch.abs(level_x - level_y).mean()
    
    return (ll_loss + hf_loss) * 0.5
