import torch
import torch.nn.functional as F

# === Standard Hinge GAN losses + R3GAN ===

def merged_d_losses(d, x_fake, x_real, sigma=0.01):
    fake_scores = d(x_fake.detach())
    real_scores = d(x_real.detach())

    fake_scores_noisy = d((x_fake + sigma*torch.randn_like(x_fake)).detach())
    real_scores_noisy = d((x_real + sigma*torch.randn_like(x_real)).detach())

    r1_penalty = (fake_scores_noisy - fake_scores).pow(2).mean()
    r2_penalty = (real_scores_noisy - real_scores).pow(2).mean()
    disc_loss = F.relu(1 + fake_scores).mean() + F.relu(1 - real_scores).mean()

    return r1_penalty, r2_penalty, disc_loss

def d_loss(d, x_fake, x_real):
    fake_scores = d(x_fake.detach())
    real_scores = d(x_real.detach())

    disc_loss = F.relu(1 + fake_scores).mean() + F.relu(1 - real_scores).mean()
    return disc_loss

def g_loss(d, x_fake):
    fake_scores = d(x_fake)
    gan_loss = -fake_scores.mean()
    return gan_loss

# === reconstruction GAN losses ===
def rec_d_loss(d, x_fake, x_real):
    s_rf = d(torch.cat([x_real, x_fake], dim=1)) # +1
    s_fr = d(torch.cat([x_fake, x_real], dim=1)) # -1

    s_rf = s_rf.mean(dim=list(range(1, s_rf.ndim)))
    s_fr = s_fr.mean(dim=list(range(1, s_fr.ndim)))

    loss = F.relu(1.0 - s_rf).mean() + F.relu(1.0 + s_fr).mean()
    return loss

def rec_g_loss(d, x_fake, x_real):
    s_rf = d(torch.cat([x_real, x_fake], dim=1)) # +1
    s_fr = d(torch.cat([x_fake, x_real], dim=1)) # -1

    s_rf = s_rf.mean(dim=list(range(1, s_rf.ndim)))
    s_fr = s_fr.mean(dim=list(range(1, s_fr.ndim)))

    return (s_rf - s_fr).mean()