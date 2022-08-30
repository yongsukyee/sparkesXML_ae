##################################################
# TRANSFORMERS
# Author: Suk Yee Yong
##################################################

import transformers
import torch
import torch.nn as nn


class ViT(nn.Module):
    
    """Vision transformer for image classification"""
    def __init__(
        self,
        input_shape,
        num_labels=2,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        patch_size=16,
        **kwargs
    ):
        super(ViT, self).__init__()
        configuration = transformers.ViTConfig(
            num_labels=num_labels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            patch_size=patch_size,
            image_size=input_shape[-1],
            num_channels=input_shape[0],
            **kwargs
        )
        self.model = transformers.ViTModel(configuration)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.model.config.hidden_size, num_labels)
        )
    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = self.classifier(outputs.last_hidden_state[:,0])
        return logits

