import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """ Attn across slices """ 
    def __init__(self, feature_dim, hidden_dim=128):
        super(AttentionModule, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # Shape of x: [bs, slices, features]
        
        # Compute attention weights
        # After this, attention_weights will have shape [bs, slices, 1]
        attention_weights = self.attention(x)
        
        # Apply softmax on slices dimension
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        # After this, output will have shape [bs, features]
        output = (attention_weights * x).sum(dim=1)
        
        return output

    def get_highest_attention_slice(self, x):
        ''' extract which slice is '''
        attention_weights = self.attention(x)
        highest_attention_indices = torch.argmax(attention_weights, dim=1)
        return highest_attention_indices
    

class MLPModule(nn.Module):
    def __init__(self, feature_dim, num_slices, hidden_dim=128, dropout=0):
        super(MLPModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim*num_slices, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Shape of x: [bs, slices, features]
        # Reshape to [bs, slice * features]
        return self.mlp(x.view(x.size(0), -1))
        

class SliceAttentionModel(nn.Module):
    def __init__(self, base_model, num_slices, num_features, num_labels, 
            hidden_dim=128, pool='mean', slice_agg='attn', regression=False, base_model_name='swin',
            dropout=0):
        """
        -slice_agg: aggregation module across slices: [attn, mlp]
        """
        super(SliceAttentionModel, self).__init__()
        self.base_model = base_model
        self.num_slices = num_slices
        if slice_agg == 'attn':
            self.aggregation = AttentionModule(num_features, hidden_dim)
            self.fc = nn.Linear(num_features, num_labels) 
        elif slice_agg == 'mlp':
            self.aggregation = MLPModule(num_features, num_slices, hidden_dim, dropout)
            self.fc = nn.Linear(hidden_dim, num_labels)   
        else:
            raise ValueError(f'{slice_agg} not among valid choices.')
        self.base_model_name = base_model_name
        if base_model_name == 'swin': 
            print(f'Using pooling for swin')
            self.pool_fn = F.adaptive_avg_pool2d if pool == 'mean' else F.adaptive_max_pool2d   
        else:
            print(f'NOT using pooling') 
        #self.final_act = nn.Softmax(dim=-1) if not regression else nn.Identity

    def forward(self, x):
        # x: tensor of shape (batch_size*num_slices, 3, 224, 224)
        x = self.base_model(x)  # Replace with the appropriate method to get features instead of predictions, e.g., 'base_model.features(x)' or 'base_model.encoder(x)' 
        # Reshape features to (batch_size, num_slices, num_features)
        #print(f'after base model: {x.shape=}')
        # Pooling
        if self.base_model_name == 'swin':
            x = self.pool_fn( x.permute(0, 3, 1, 2), 1).squeeze(-1).squeeze(-1)
            # print(f'after pooling: {x.shape=}')
   
        # reshape to get slices again:
        x = x.view(x.size(0) // self.num_slices, self.num_slices, -1)
        # print(f'After reshape: {x.shape=}')

        x  = self.aggregation(x)
        out = self.fc(x)
        return out #self.final_act(out)

    def freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2): # You can adjust `num_classes` accordingly.
        super(Simple3DCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1), # Output: [bs, 32, 182, 224, 224]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: [bs, 32, 91, 112, 112]

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), # Output: [bs, 64, 91, 112, 112]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: [bs, 64, 45, 56, 56]

            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1), # Output: [bs, 128, 45, 56, 56]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Output: [bs, 64, 22, 28, 28]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 28 * 28, 256), # Adjust this size accordingly
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

class Residual3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Residual3DCNN, self).__init__()
        self.act_fn = nn.ReLU()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(16)

        self.adjust_channel_1_to_16 = nn.Conv3d(1, 16, kernel_size=1)
        self.adjust_channel_16_to_32 = nn.Conv3d(16, 32, kernel_size=1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(275968, 128), 
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            #nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # First block with residual connection
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)
       
        #print(f'Before adjustment 1: {out.shape=}, {identity.shape=}')
        if identity.size(1) == 1:        
            identity = self.adjust_channel_1_to_16(identity)
        #print(f'{out.shape=}, {identity.shape=}')
        out = out + identity
        out = nn.MaxPool3d(kernel_size=2, stride=2)(out)

        # Second block with residual connection
        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_fn(out)
        
        if identity.size(1) == 16:        
            identity = self.adjust_channel_16_to_32(identity)
        #print(f'{out.shape=}, {identity.shape=}')
        out = out + identity
        out = nn.MaxPool3d(kernel_size=2, stride=2)(out)

        # Third block
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act_fn(out)
  
        out = nn.MaxPool3d(kernel_size=2, stride=2)(out)

        out = self.classifier(out)
        return out

# Load the pretrained model
def get_pretrained_slowfast(num_classes: int):
    # Create the SlowFast model
    model_name = "slowfast_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    
    return model 
    
    ## Update the classification head for your number of classes
    #model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, num_classes)
    #
    #return model

if __name__ == "__main__":
	model = get_pretrained_slowfast(num_classes=2)
	from IPython import embed; embed()

