import torch

def approximate_r1_loss(
    discriminator,
    latents,
    sigma=0.01,
    Lambda=100.0,
):
    """
    Approx. R1 via finite differences:
        L_{aR1} = || D(x) - D(x + noise) ||^2
    """
    noise = sigma * torch.randn_like(latents)
    d_real = discriminator(latents)
    d_noisy = discriminator(latents + noise)
    return ((d_real - d_noisy).pow(2).mean()) * Lambda


def approximate_r2_loss(
    discriminator,
    fake_images,
    sigma=0.01,
    Lambda=100.0,

):
    """
    Approx. R2 via finite differences:
        L_{aR2} = || D(x_fake) - D(x_fake + noise) ||^2
    """
    noise = sigma * torch.randn_like(fake_images)
    d_fake = discriminator(fake_images)
    d_fake_noisy = discriminator(fake_images + noise)
    return ((d_fake - d_fake_noisy).pow(2).mean()) * Lambda

def gan_loss_with_approximate_penalties(
    discriminator,
    images,
    reconstructions,
    discriminator_turn=True,
    sigma=0.01,
    Lambda=100.0
):
    """
    Non saturating Relativistic GAN loss of the form:
        E_{z,x}[ f(-(D(G(z)) - D(x))) ].
    for the discriminator, and form:
        E_{z,x}[ f(-(D(x) - D(G(z)))) ].
    for the generator.

    Adds approximate R1 and R2 penalties to the discriminator loss.

    Args:
        discriminator: Discriminator network D
        images: A batch of real data (x)
        reconstructions: A batch of fake data (G(z))
        discriminator_turn: If True, calculates loss for the discriminator, else for the generator
        sigma: Standard deviation of the noise added to the real images
        Lambda: Weight for the approximate R1 and R2 penalties
    """
    f = torch.nn.functional.softplus

    # Evaluate discriminator
    disc_real = discriminator(images)
    disc_fake = discriminator(reconstructions)

    # Compute the loss using f
    if discriminator_turn:
        loss = f(disc_fake - disc_real).mean()
        loss += approximate_r1_loss(discriminator, images, sigma, Lambda)
        loss += approximate_r2_loss(discriminator, reconstructions, sigma, Lambda)
    else:
        loss = f(disc_real - disc_fake)
    return loss