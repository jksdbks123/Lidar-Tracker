import torch

def masked_mse_loss(predictions, targets, masks):
    """
    Masked MSE loss for trajectory prediction
    
    Args:
        predictions: Predicted positions [batch_size, output_frames]
        targets: Ground truth positions [batch_size, output_frames]
        masks: Binary masks for valid positions [batch_size, output_frames]
        
    Returns:
        loss: Masked MSE loss
    """
    # Compute squared error
    squared_error = (predictions - targets) ** 2
    
    # Apply mask
    masked_error = squared_error * masks
    
    # Sum over time and average over batch
    loss = masked_error.sum(dim=1) / (masks.sum(dim=1) + 1e-6)
    
    return loss.mean()

def social_trajectory_loss(predictions, targets, confidences, masks):
    """
    Combined loss function for position prediction and confidence
    
    Args:
        predictions: Predicted positions [batch_size, output_frames]
        targets: Ground truth positions [batch_size, output_frames]
        confidences: Prediction confidences [batch_size, output_frames]
        masks: Binary masks for valid positions [batch_size, output_frames]
        
    Returns:
        loss: Combined loss value
    """
    # Basic masked MSE loss
    mse_loss = masked_mse_loss(predictions, targets, masks)
    
    # Confidence-weighted error
    weighted_error = ((predictions - targets) ** 2) * confidences * masks
    confidence_weighted_error = weighted_error.sum(dim=1) / (masks.sum(dim=1) + 1e-6)
    confidence_weighted_error = confidence_weighted_error.mean()
    
    # Confidence calibration (higher confidence for lower error)
    neg_log_conf = -torch.log(confidences + 1e-6) * masks
    error_weight = torch.exp(-5.0 * ((predictions - targets) ** 2)) * masks
    confidence_calibration = (neg_log_conf * error_weight).sum(dim=1) / (masks.sum(dim=1) + 1e-6)
    confidence_calibration = confidence_calibration.mean()
    
    # Combined loss
    loss = mse_loss + 0.5 * confidence_weighted_error + 0.2 * confidence_calibration
    
    return loss, mse_loss, confidence_weighted_error, confidence_calibration