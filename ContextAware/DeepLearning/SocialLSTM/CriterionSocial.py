import torch
import torch.nn.functional as F

def kl_divergence_loss(predictions, targets, masks):
    """
    KL divergence loss for distribution prediction
    
    Args:
        predictions: Predicted distributions [batch_size, output_frames, lane_cells]
        targets: Target distributions [batch_size, output_frames, lane_cells]
        masks: Binary masks for valid positions [batch_size, output_frames]
        
    Returns:
        loss: KL divergence loss
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    predictions = predictions + epsilon
    targets = targets + epsilon
    
    # Normalize distributions
    predictions = predictions / predictions.sum(dim=2, keepdim=True)
    targets = targets / targets.sum(dim=2, keepdim=True)
    
    # Compute KL divergence
    kl_div = targets * (torch.log(targets) - torch.log(predictions))
    kl_div = kl_div.sum(dim=2)  # Sum over lane cells
    
    # Apply mask
    masked_kl_div = kl_div * masks
    
    # Average over valid frames
    loss = masked_kl_div.sum(dim=1) / (masks.sum(dim=1) + epsilon)
    
    return loss.mean()

def js_divergence_loss(predictions, targets, masks):
    """
    Jensen-Shannon divergence loss for distribution prediction
    
    Args:
        predictions: Predicted distributions [batch_size, output_frames, lane_cells]
        targets: Target distributions [batch_size, output_frames, lane_cells]
        masks: Binary masks for valid positions [batch_size, output_frames]
        
    Returns:
        loss: JS divergence loss
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    predictions = predictions + epsilon
    targets = targets + epsilon
    
    # Normalize distributions
    predictions = predictions / predictions.sum(dim=2, keepdim=True)
    targets = targets / targets.sum(dim=2, keepdim=True)
    
    # Compute midpoint distribution
    m = 0.5 * (predictions + targets)
    
    # Compute KL divergences
    kl_p_m = predictions * (torch.log(predictions) - torch.log(m))
    kl_p_m = kl_p_m.sum(dim=2)  # Sum over lane cells
    
    kl_t_m = targets * (torch.log(targets) - torch.log(m))
    kl_t_m = kl_t_m.sum(dim=2)  # Sum over lane cells
    
    # Jensen-Shannon divergence
    js_div = 0.5 * (kl_p_m + kl_t_m)
    
    # Apply mask
    masked_js_div = js_div * masks
    
    # Average over valid frames
    loss = masked_js_div.sum(dim=1) / (masks.sum(dim=1) + epsilon)
    
    return loss.mean()

def expected_position_loss(predictions, target_positions, masks):
    """
    MSE loss between expected position from distribution and target position
    
    Args:
        predictions: Predicted distributions [batch_size, output_frames, lane_cells]
        target_positions: Target positions [batch_size, output_frames]
        masks: Binary masks for valid positions [batch_size, output_frames]
        
    Returns:
        loss: MSE loss of expected positions
    """
    batch_size = predictions.shape[0]
    output_frames = predictions.shape[1]
    lane_cells = predictions.shape[2]
    device = predictions.device
    
    # Create positions tensor
    positions = torch.arange(0, lane_cells, device=device).float()
    
    # Compute expected positions from predicted distributions
    expected_positions = torch.sum(predictions * positions.unsqueeze(0).unsqueeze(0), dim=2)
    
    # Compute squared error
    squared_error = (expected_positions - target_positions) ** 2
    
    # Apply mask
    masked_error = squared_error * masks
    
    # Average over valid frames
    loss = masked_error.sum(dim=1) / (masks.sum(dim=1) + 1e-8)
    
    return loss.mean()

def combined_distribution_loss(predictions, targets, target_positions, masks, alpha=0.5, beta=0.5):
    """
    Combined loss function for distribution prediction
    
    Args:
        predictions: Predicted distributions [batch_size, output_frames, lane_cells]
        targets: Target distributions [batch_size, output_frames, lane_cells]
        target_positions: Target positions [batch_size, output_frames]
        masks: Binary masks for valid positions [batch_size, output_frames]
        alpha: Weight for JS divergence loss
        beta: Weight for expected position loss
        
    Returns:
        loss: Combined loss
        js_loss: JS divergence loss component
        pos_loss: Expected position loss component
    """
    # JS divergence loss
    js_loss = js_divergence_loss(predictions, targets, masks)
    
    # Expected position loss
    pos_loss = expected_position_loss(predictions, target_positions, masks)
    
    # Combined loss
    loss = alpha * js_loss + beta * pos_loss
    
    return loss, js_loss, pos_loss


def focal_loss(predictions, target_positions, masks, gamma=1.0, alpha=0.5):
    """
    Focal loss for lane cell classification
    
    Args:
        predictions: Logits or probability distributions [batch_size, output_frames, lane_cells]
        target_positions: Target positions as indices [batch_size, output_frames]
        masks: Binary masks for valid positions [batch_size, output_frames]
        gamma: Focusing parameter
        alpha: Class weight parameter
        
    Returns:
        loss: Masked focal loss
    """
    batch_size = predictions.shape[0]
    output_frames = predictions.shape[1]
    lane_cells = predictions.shape[2]
    device = predictions.device
    
    # Convert target positions to integer indices if they're not already
    target_positions = target_positions.long()
    
    # Convert to one-hot encoding
    target_one_hot = torch.zeros(batch_size, output_frames, lane_cells, device=device)
    for b in range(batch_size):
        for t in range(output_frames):
            if masks[b, t] > 0.5:
                pos = target_positions[b, t]
                if 0 <= pos < lane_cells:  # Ensure position is valid
                    target_one_hot[b, t, pos] = 1.0
    
    # Apply softmax to get probabilities
    probs = F.softmax(predictions, dim=-1)
    
    # Compute focal loss
    pt = torch.sum(probs * target_one_hot, dim=-1)  # Probability of the target class
    focal_weight = alpha * (1 - pt) ** gamma
    
    # Compute cross entropy loss
    log_pt = torch.log(pt + 1e-10)  # Add small epsilon to avoid log(0)
    loss = -focal_weight * log_pt
    
    # Apply mask
    masked_loss = loss * masks
    
    # Average over valid frames
    final_loss = masked_loss.sum(dim=1) / (masks.sum(dim=1) + 1e-8)
    
    return final_loss.mean()

import numpy as np

def evaluate_prediction_metrics(pred_distributions, target_positions, output_masks):
    """
    Calculate practical metrics for trajectory prediction performance
    
    Args:
        pred_distributions: Predicted distributions [batch_size, output_frames, lane_cells]
        target_positions: Target positions [batch_size, output_frames]
        output_masks: Masks for valid positions [batch_size, output_frames]
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Move everything to CPU for numpy operations
    pred_distributions = pred_distributions.detach().cpu().numpy()
    target_positions = target_positions.detach().cpu().numpy()
    output_masks = output_masks.detach().cpu().numpy()
    
    batch_size, output_frames, lane_cells = pred_distributions.shape
    
    # Calculate most likely positions (argmax)
    argmax_positions = np.argmax(pred_distributions, axis=2)
    
    # Calculate expected positions (weighted average)
    cell_positions = np.arange(lane_cells)
    expected_positions = np.sum(pred_distributions * cell_positions.reshape(1, 1, -1), axis=2)
    
    # Initialize metric arrays
    argmax_errors = np.zeros((batch_size, output_frames))
    expected_errors = np.zeros((batch_size, output_frames))
    
    # Calculate errors for each prediction
    for b in range(batch_size):
        for t in range(output_frames):
            if output_masks[b, t] > 0.5:  # Only consider valid positions
                # Error using argmax position
                argmax_errors[b, t] = abs(argmax_positions[b, t] - target_positions[b, t])
                
                # Error using expected position
                expected_errors[b, t] = abs(expected_positions[b, t] - target_positions[b, t])
    
    # Calculate accuracy (within tolerance)
    tolerance_1 = np.mean((argmax_errors <= 1) * output_masks) / np.mean(output_masks)
    tolerance_3 = np.mean((argmax_errors <= 3) * output_masks) / np.mean(output_masks)
    tolerance_5 = np.mean((argmax_errors <= 5) * output_masks) / np.mean(output_masks)
    
    # Calculate mean and median errors
    valid_mask = output_masks > 0.5
    valid_argmax_errors = argmax_errors[valid_mask]
    valid_expected_errors = expected_errors[valid_mask]
    
    # Calculate metrics
    metrics = {
        'mean_argmax_error': float(np.mean(valid_argmax_errors)),
        'median_argmax_error': float(np.median(valid_argmax_errors)),
        'mean_expected_error': float(np.mean(valid_expected_errors)),
        'median_expected_error': float(np.median(valid_expected_errors)),
        'accuracy_tol_1': float(tolerance_1),
        'accuracy_tol_3': float(tolerance_3),
        'accuracy_tol_5': float(tolerance_5)
    }
    
    # Calculate error by prediction horizon
    horizon_errors = []
    for t in range(output_frames):
        valid_t = output_masks[:, t] > 0.5
        if np.sum(valid_t) > 0:
            mean_error_t = np.mean(argmax_errors[valid_t, t])
            horizon_errors.append(mean_error_t)
        else:
            horizon_errors.append(np.nan)
    
    metrics['horizon_errors'] = horizon_errors
    
    return metrics