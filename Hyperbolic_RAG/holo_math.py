import torch

def clamp_norm(x, max_norm=1.0 - 1e-5):
    """Clamp the norm of a tensor to prevent numerical instability at the boundary of the Poincaré ball."""
    norm = torch.norm(x, dim=-1, keepdim=True)
    cond = norm > max_norm
    clamped_norm = torch.clamp(norm, max=max_norm)
    # Avoid division by zero when norm is 0
    norm_safe = torch.clamp(norm, min=1e-15)
    return torch.where(cond, x / norm_safe * clamped_norm, x)

def poincare_distance(x, y):
    """
    Calculate the Poincaré (geodesic) distance between two points x and y.
    d_H(x, y) = arcosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2) * (1 - ||y||^2)))
    """
    x_clamped = clamp_norm(x)
    y_clamped = clamp_norm(y)
    
    sqdist = torch.sum((x_clamped - y_clamped) ** 2, dim=-1)
    sqnorm_x = torch.sum(x_clamped ** 2, dim=-1)
    sqnorm_y = torch.sum(y_clamped ** 2, dim=-1)
    
    denominator = (1 - sqnorm_x) * (1 - sqnorm_y)
    # Add epsilon to denominator to avoid division by zero
    denominator = torch.clamp(denominator, min=1e-15)
    
    argument = 1 + 2 * sqdist / denominator
    
    # Add epsilon to arcosh argument to prevent NaN at exactly 1
    argument = torch.clamp(argument, min=1.0 + 1e-7)
    
    return torch.acosh(argument)

def mobius_addition(x, y, c=1.0):
    """
    Möbius addition of x and y in the Poincaré ball.
    """
    x_clamped = clamp_norm(x)
    y_clamped = clamp_norm(y)
    
    inner_xy = torch.sum(x_clamped * y_clamped, dim=-1, keepdim=True)
    sqnorm_x = torch.sum(x_clamped ** 2, dim=-1, keepdim=True)
    sqnorm_y = torch.sum(y_clamped ** 2, dim=-1, keepdim=True)
    
    num = (1 + 2 * c * inner_xy + c * sqnorm_y) * x_clamped + (1 - c * sqnorm_x) * y_clamped
    den = 1 + 2 * c * inner_xy + c**2 * sqnorm_x * sqnorm_y
    den = torch.clamp(den, min=1e-15)
    
    return clamp_norm(num / den)

def exp_map_0(v):
    """
    Exponential map at the origin: projects Euclidean vector v to the Poincaré ball.
    exp_0(v) = tanh(||v||) * (v / ||v||)
    """
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    norm_v_safe = torch.clamp(norm_v, min=1e-15)
    
    direction = v / norm_v_safe
    magnitude = torch.tanh(norm_v)
    
    # If norm_v is very close to 0, return v directly
    cond = norm_v < 1e-15
    return torch.where(cond, v, magnitude * direction)
