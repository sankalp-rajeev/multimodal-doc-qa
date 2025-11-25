"""
Multi-Task LayoutLMv3: Simultaneous Span Extraction + BIO Tagging
"""

import torch
import torch.nn as nn
from transformers import LayoutLMv3Model, LayoutLMv3PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class LayoutLMv3ForMultiTask(LayoutLMv3PreTrainedModel):
    """
    LayoutLMv3 with two task heads:
    1. Question Answering (span extraction)
    2. Token Classification (BIO tagging)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = config.num_labels
        
        # Shared encoder
        self.layoutlmv3 = LayoutLMv3Model(config)
        
        # Task Head 1: QA (span extraction)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        
        # Task Head 2: BIO tagging
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        pixel_values=None,
        start_positions=None,
        end_positions=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Run shared encoder
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Task A: QA
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        # Task B: BIO
        bio_logits = self.classifier(sequence_output)
        
        # Compute losses
        total_loss = None
        if start_positions is not None and end_positions is not None and labels is not None:
            # QA loss
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct_qa = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct_qa(start_logits, start_positions)
            end_loss = loss_fct_qa(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            
            # print(f"DEBUG: bio_logits shape: {bio_logits.shape}")
            # print(f"DEBUG: labels shape: {labels.shape}")
            # print(f"DEBUG: After view - logits: {bio_logits.view(-1, self.num_labels).shape}")
            # print(f"DEBUG: After view - labels: {labels.view(-1).shape}")
            
            # BIO loss - ignore_index=-100 handles padding automatically
            seq_len = labels.size(1)
            bio_logits_truncated = bio_logits[:, :seq_len, :]
            
            loss_fct_bio = nn.CrossEntropyLoss(ignore_index=-100)
            bio_loss = loss_fct_bio(
                bio_logits_truncated.reshape(-1, self.num_labels),
                labels.reshape(-1)
            )
            
            # Combine losses
            total_loss = qa_loss + bio_loss
        
        if not return_dict:
            output = (start_logits, end_logits, bio_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )