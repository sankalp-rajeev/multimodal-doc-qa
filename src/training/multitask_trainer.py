"""
Custom Trainer for Multi-Task Learning
"""

from transformers import Trainer
import torch


class MultiTaskTrainer(Trainer):
    """
    Custom trainer that handles multiple task outputs
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for multi-task model
        """
        # Forward pass
        outputs = model(**inputs)
        
        # Loss is already computed in the model's forward()
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            logits = (outputs.start_logits, outputs.end_logits)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Get labels
        labels = (inputs.get("start_positions"), inputs.get("end_positions"))
        
        return (loss, logits, labels)