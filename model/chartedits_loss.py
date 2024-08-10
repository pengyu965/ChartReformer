import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ChartEdits_CrossEntropy(nn.Module):
    def __init__(self):
        super(ChartEdits_CrossEntropy, self).__init__()

    def forward(self, model_out,input_batch,diff_idx):
        labels = input_batch["labels"]
        # Convert diff_idx to boolean mask if it's not already
        change_mask = diff_idx.bool()
        remain_mask = ~change_mask
        
        # Calculate loss for the parts to be changed
        loss = F.cross_entropy(model_out.logits.contiguous().view(-1, model_out.logits.size(-1)), 
                               labels.contiguous().view(-1), 
                               reduction='none')

        if change_mask.sum() == 0 or remain_mask.sum() == 0:
            return loss.mean()
        
        loss_change = torch.sum(loss*change_mask.contiguous().view(-1))/change_mask.sum()
        loss_remain = torch.sum(loss*remain_mask.contiguous().view(-1))/remain_mask.sum()

        
        # Calculate the mean of the two losses
        total_loss = 2/(1/loss_change + 1/loss_remain)

        
        return total_loss
    