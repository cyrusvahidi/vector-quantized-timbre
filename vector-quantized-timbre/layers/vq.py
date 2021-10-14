import gin, torch, torch.nn as nn, torch.nn.functional as F


class VQEmbedding(nn.Module):

    def __init__(
        self, 
        num_embeddings=1024, 
        embedding_dim=128, 
        commitment_cost=0.25, 
        use_codebook_loss=True, 
        axis=-1
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._commitment_cost = commitment_cost
        self._use_codebook_loss = use_codebook_loss
        self._axis = axis

        self.num_embeddings = num_embeddings

    def forward(self, input, verbose=False):
        if self._axis != -1:
            input = input.transpose(self._axis, -1)

        distances = (torch.sum(input ** 2, axis=-1, keepdim=True)
                     - 2 * torch.matmul(input, self.embedding.weight.T)
                     + torch.sum(self.embedding.weight ** 2, axis=-1))
        ids = torch.argmin(distances, axis=-1)
        quantized = self.embedding(ids)

        encodings = torch.zeros(ids.shape[0] * ids.shape[1], self.num_embeddings)
        encodings.scatter_(1, ids.cpu().reshape(-1, 1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        losses = {
            'commitment': self._commitment_cost*((quantized.detach() - input) ** 2).mean(axis=-1)
        }

        losses['codebook'] = ((quantized - input.detach()) ** 2).mean(axis=-1)

        quantized = (quantized - input).detach() + input

        if self._axis != -1:
            quantized = quantized.transpose(self._axis, -1).contiguous()

        return quantized, ids, losses, perplexity


@gin.configurable
class VQEmbeddingEMA(nn.Module):
    def __init__( 
        self,
        decay=0.99, 
        epsilon=1e-5,
        num_embeddings=1024, 
        embedding_dim=128,
        commitment_cost=0.25
    ):
        super(VQEmbeddingEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))

        self.ema_w.data.normal_()

    def forward(self, x):
        input_shape = x.shape
        flat_input = x.reshape(-1, self.embedding_dim)
        distances = (torch.sum(flat_input ** 2, axis=-1, keepdim=True)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.T)
                     + torch.sum(self.embedding.weight ** 2, axis=-1))

        encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        loss = self.commitment_cost * e_latent_loss
        
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, encoding_indices, loss, perplexity