import torch
import torch.nn.functional as F
import pytorch_wavelets as pw
import math

def dwt_loss_fn(x, y):
    """
    Compute DWT loss between two images using L1 distance between wavelet coefficients

    Args:
        x: First image tensor [B,C,H,W]
        y: Second image tensor [B,C,H,W]

    Returns:
        Average L1 loss between wavelet coefficients
    """
    orig_dtype = x.dtype

    # Disable autocast to ensure clean float32 computation
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # Create wavelet transform with 4 levels
        xfm = pw.DWTForward(J=4, wave='haar').to(x.device)

        # Get coefficients for both images in float32
        # LL is in yl, (LH,HL,HH) are in yh
        yl_x, yh_x = xfm(x.float())
        yl_y, yh_y = xfm(y.float())

        # L1 loss on low frequency component (LL)
        ll_loss = torch.abs(yl_x - yl_y).mean()

        # L1 loss on high frequency components (LH,HL,HH) at each level
        hf_loss = 0.
        for level_x, level_y in zip(yh_x, yh_y):
            hf_loss += torch.abs(level_x - level_y).mean()

        loss = (ll_loss + hf_loss) * 0.5

    # Cast the final result back to original dtype
    return loss.to(orig_dtype)

def _pad_to_even_3d(x: torch.Tensor):
    # x: [B,C,T,H,W]
    B,C,T,H,W = x.shape
    pad_t = T % 2
    pad_h = H % 2
    pad_w = W % 2
    if pad_t or pad_h or pad_w:
        # F.pad expects (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
    return x

def _haar3d_subbands(x: torch.Tensor):
    """
    One-level 3D Haar DWT using conv3d.
    Input:
      x: [B,C,T,H,W]
    Returns:
      low:   [B,C,T/2,H/2,W/2]    (LLL)
      highs: [B,C,7,T/2,H/2,W/2]  (all high bands)
    """
    x = _pad_to_even_3d(x)
    B,C,T,H,W = x.shape
    device, dtype = x.device, x.dtype

    s = 1.0 / math.sqrt(2.0)
    lo = torch.tensor([s, s], device=device, dtype=dtype)
    hi = torch.tensor([s, -s], device=device, dtype=dtype)

    kernels = []
    for ft in (lo, hi):
        for fh in (lo, hi):
            for fw in (lo, hi):
                k = ft[:, None, None] * fh[None, :, None] * fw[None, None, :]
                kernels.append(k)
    K = torch.stack(kernels, dim=0)   # [8,2,2,2]
    K = K[:, None]                    # [8,1,2,2,2]

    weight = K.repeat(C, 1, 1, 1, 1)  # [8*C,1,2,2,2]

    out = F.conv3d(x, weight, stride=2, groups=C)
    out = out.view(B, C, 8, out.shape[-3], out.shape[-2], out.shape[-1])

    low = out[:, :, 0]
    highs = out[:, :, 1:]
    return low, highs

def dwt_loss_fn_3d(x: torch.Tensor, y: torch.Tensor,
                   J: int = 1,
                   w_lll: float = 1.0,
                   w_high: float = 1.0):
    """
    3D Haar DWT loss for videos in [B,T,C,H,W]
    """
    orig_dtype = x.dtype
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # permute to [B,C,T,H,W]
        x0 = x.permute(0, 2, 1, 3, 4).float()
        y0 = y.permute(0, 2, 1, 3, 4).float()

        loss = 0.0
        for _ in range(J):
            low_x, high_x = _haar3d_subbands(x0)
            low_y, high_y = _haar3d_subbands(y0)

            loss_lll = (low_x - low_y).abs().mean()
            loss_high = (high_x - high_y).abs().mean()

            loss = loss + (w_lll * loss_lll + w_high * loss_high)

            x0, y0 = low_x, low_y

        loss = loss / J

    return loss.to(orig_dtype)


def test_dwt_loss_fn_3d():
    x = torch.randn(1, 16, 3, 180, 320).cuda()
    y = torch.randn(1, 16, 3, 180, 320).cuda()
    loss = dwt_loss_fn_3d(x, y)
    print(loss)

if __name__ == "__main__":
    test_dwt_loss_fn_3d()