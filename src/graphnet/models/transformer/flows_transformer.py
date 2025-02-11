import math
import torch
import torch.nn as nn
from graphnet.models.components.embedding import SinusoidalPosEmb
from graphnet.models.gnn.gnn import GNN

from torch_geometric.data import Data
from torch import Tensor
from graphnet.models.utils import array_to_sequence
from torch_geometric.utils import to_dense_batch

class FlowsTransformer(GNN):
    def __init__(self, num_features=21, num_pos_features = 10, num_value_features = 12, output_dim=128, d_model=100, num_layers=20, nhead=1,
                 dim_feedforward=512, dropout=0, emb_dim=16, scaling_factor=4096, n_freq=10000):
        """
        Args:
            num_features: Number of continuous input features per token.
            num_pos_features: Number of features used for positional encoding.
            num_value_features: Number of features concatenated to the positional encoding.
            output_dim: Dimension of the model output.
            d_model: Internal transformer model dimension.
            num_layers: Number of transformer encoder layers.
            nhead: Number of attention heads.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout probability.
            emb_dim: Embedding dimension for each feature (should be even).
            scaling_factor: Factor to multiply the normalized features before encoding. 4096 seems to be common.
            n_freq: Frequency parameter for the Fourier embedding.
        """
        super().__init__(num_features, output_dim)
        self.num_features = num_features
        self.num_pos_features = num_pos_features
        self.num_value_features = num_value_features
        self.emb_dim = emb_dim  # must be even!!
        self.scaling_factor = scaling_factor
        self.n_freq = n_freq

        # Projection layer from concatenated Fourier embeddings to d_model.
        self.input_proj = nn.Linear(self.num_pos_features*self.emb_dim+self.num_value_features, d_model)
        
        self.fourier_embed = SinusoidalPosEmb(dim=self.emb_dim)

        # Trasnformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=torch.nn.LeakyReLU())
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)


    def forward(self, data: Data) -> Tensor:
        
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, num_features]
               All features are assumed to be normalized (e.g. to [-1,1]).
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        #x, _, _ = array_to_sequence(data.x, data.batch, padding_value=0)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Padding due to iregular length
        x, mask = to_dense_batch(x, batch)
        
        # Positional features for embedding
        positional_features = x[:, :, :10]
        #print(f"Positional features shape: {positional_features.shape}")
        # Sinusoidal positional encoding
        x_emb = self.fourier_embed(positional_features * self.scaling_factor)  
        
        # Reshape
        x_emb = x_emb.view(x_emb.shape[0], x_emb.shape[1], -1)
        #print(f"Embdedding shape: {x_emb.shape}")
        # Rest of features
        node_features = x[:, :, [3,10,11,12,13,14,15,16,17,18,19,20]]
        #print(f"Node features shape: {node_features.shape}")        
        # Concatenate positional and other features
        x_emb = torch.cat((x_emb, node_features), dim=-1)
        #print(f"Concatenated features shape: {x_emb.shape}")
        
        # Project to transformer input space
        x_proj = self.input_proj(x_emb) * math.sqrt(x_emb.shape[-1])

        # Reshape for transformer
        x_proj = x_proj.transpose(0, 1)

        # Pass through transformer encoder
        x_encoded = self.transformer_encoder(x_proj, src_key_padding_mask=~mask)
        #print(x_encoded.shape)
        # Mean pooling over sequence dimension
        x_pooled = x_encoded.mean(dim=0)  # Shape: [batch_size, d_model]

        x_out = self.fc_out(x_pooled)
        print(x_out.max().item(),x_out.mean().item(),x_out.min().item())
        # Final output layer
        return x_out 

