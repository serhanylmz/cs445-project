import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class TopicGroupedAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        return attention_output

class TGANet(nn.Module):
    def __init__(self, num_labels=3, hidden_dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size
        
        self.topic_attention = TopicGroupedAttention(self.hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        
        # Apply topic-grouped attention
        attended_output = self.topic_attention(sequence_output, attention_mask.unsqueeze(1).unsqueeze(2))
        
        # Pool the output (use [CLS] token representation)
        pooled_output = attended_output[:, 0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return loss, logits
        
        return logits

class BaselineBERTClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return loss, logits
        
        return logits 