---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Transformer

## Token Embedding

```python
class TokenEmbedding(nn.Module):
    """Convert token indices to embeddings."""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

## Position Encoding

### Learned
```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
```

### Sinusoidal

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
```

### RoPE

```python
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Precompute the frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, q, k):
        # q, k shape: (batch_size, seq_len, n_heads, d_head)
        batch_size, seq_len, n_heads, d_head = q.shape
        
        # Generate position indices
        pos = torch.arange(seq_len, device=q.device).float()
        
        # Compute rotation angles
        # einsum('i,j->ij', pos, self.inv_freq) does outer product:
        # - 'i' refers to dimension of pos (seq_len)
        # - 'j' refers to dimension of inv_freq (d_model/2)
        # - 'ij' means output has both dimensions
        # This is equivalent to: pos.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        # or simply: pos[:, None] * self.inv_freq[None, :]
        
        # Using einsum (original):
        sincos = torch.einsum('i,j->ij', pos, self.inv_freq)
        
        # Alternative without einsum (clearer for learning):
        # sincos = pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        # This creates a matrix where sincos[i,j] = pos[i] * inv_freq[j]
        
        sin = sincos.sin()[None, :, None, :]
        cos = sincos.cos()[None, :, None, :]
        
        # Apply rotary encoding to q and k
        q_rot = self.apply_rotary(q, sin, cos)
        k_rot = self.apply_rotary(k, sin, cos)
        
        return q_rot, k_rot
    
    def apply_rotary(self, x, sin, cos):
        # Split the last dimension into pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Apply rotation
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        # Concatenate back
        y = torch.stack([y1, y2], dim=-1).flatten(-2)
        return y
```

## Self Attention

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, causal=False):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(scores.device)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)
```

## Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, causal=False):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(scores.device)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_O(output)
```

## Feed-Forward Network

```python
class FeedFowardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model, bias=True)
        )
        
    def forward(self, X):
        return self.ffn(X)
```

## Normalization

### LayerNorm
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        # Calculate mean and std along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta
```

### BatchNorm
```python
class BatchNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(d_model))
        self.register_buffer('running_var', torch.ones(d_model))
        self.training = True
        
    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=(0, 1))
            batch_var = x.var(dim=(0, 1), unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize using batch statistics
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta
```

## Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        
        self.mha = MultiHeadSelfAttention(d_model=d_model, n_heads=self.n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedFowardNetwork(d_model=d_model,
                                     d_ffn=self.d_ffn)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None, causal=False):
        X += self.dropout(self.mha(X, mask, causal))
        X = self.ln1(X)
        X += self.dropout(self.ffn(X))
        X = self.ln2(X)
        return X
```

## Transformer
```python
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        mode='encoder',  # 'encoder' or 'decoder'
        pos_encoding='sinusoidal',  # 'sinusoidal', 'learned', or 'rotary'
        norm_type='layer'
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        # Positional encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        elif pos_encoding == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_len)
        elif pos_encoding == 'rotary':
            self.pos_encoding = None  # RoPE is applied in attention
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        if norm_type == 'layer':
            self.final_norm = LayerNorm(d_model)
        else:
            self.final_norm = BatchNorm(d_model)
            
        # Output projection for language modeling
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Token embedding
        x = self.token_embedding(x)
        
        # Add positional encoding (if not using rotary)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            # Use causal masking for decoder mode
            x = block(x, mask, causal=(self.mode == 'decoder'))
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary size
        logits = self.output_projection(x)
        
        return logits
```

# Decoding
## Greedy
```python
def greedy_decode(logits: torch.Tensor) -> int:
    return torch.argmax(logits, dim=-1).item()
```

## Temperature
```python
def temperature_decode(logits: torch.Tensor, temperature: float = 1.0) -> int:
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

## Top K
```python
def top_k_decode(logits: torch.Tensor, temperature: float = 1.0, k: int = 50) -> int:
    top_k_logits, top_k_indices = torch.topk(logits, k=min(k, logits.size(-1)))
    probs = F.softmax(top_k_logits / temperature, dim=-1)
    selected_idx = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[selected_idx].item()
```

## Top P
```python
def top_p_decode(logits: torch.Tensor, temperature: float = 1.0, p: float = 0.9) -> int:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_idx = torch.where(cumsum_probs > p)[0]
    if len(cutoff_idx) > 0:
        # Keep at least 1 token
        cutoff_idx = max(1, cutoff_idx[0].item())
    else:
        cutoff_idx = len(sorted_probs)
    
    # Keep only top-p tokens
    top_p_probs = sorted_probs[:cutoff_idx]
    top_p_indices = sorted_indices[:cutoff_idx]
    
    # Renormalize and sample
    top_p_probs = top_p_probs / top_p_probs.sum()
    selected_idx = torch.multinomial(top_p_probs, num_samples=1).item()
    
    return top_p_indices[selected_idx].item()
```

## Beam Search
```python
class BeamSearch:
    def __init__(self, beam_width: int = 5, max_length: int = 100):
        self.beam_width = beam_width
        self.max_length = max_length
    
    def search(self, initial_logits: torch.Tensor, 
               get_next_logits_fn, # Function that takes token sequence and returns next logits
               start_token: int = 0, 
               end_token: int = 1) -> List[Tuple[List[int], float]]:
        """
        Perform beam search decoding
        
        Args:
            initial_logits: Logits for first token (shape: [vocab_size])
            get_next_logits_fn: Function that takes a sequence and returns next logits
            start_token: Token to start sequences with
            end_token: Token that ends sequences
            
        Returns:
            List of (sequence, score) tuples, sorted by score (highest first)
        """
        # Initialize beams: (sequence, log_prob)
        beams = [([start_token], 0.0)]
        completed_sequences = []
        
        # Convert initial logits to log probabilities
        log_probs = F.log_softmax(initial_logits, dim=-1)
        
        # Get top beam_width initial tokens
        top_log_probs, top_indices = torch.topk(log_probs, self.beam_width)
        
        # Initialize beams with top tokens
        beams = []
        for i in range(self.beam_width):
            token = top_indices[i].item()
            log_prob = top_log_probs[i].item()
            beams.append(([start_token, token], log_prob))
        
        # Generate sequences
        for step in range(self.max_length - 1):
            if not beams:
                break
                
            candidates = []
            
            for sequence, current_score in beams:
                # Skip if sequence already ended
                if sequence[-1] == end_token:
                    completed_sequences.append((sequence, current_score))
                    continue
                
                # Get next token probabilities
                next_logits = get_next_logits_fn(sequence)
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                
                # Add all possible next tokens
                for token_id in range(next_log_probs.size(0)):
                    new_sequence = sequence + [token_id]
                    new_score = current_score + next_log_probs[token_id].item()
                    candidates.append((new_sequence, new_score))
            
            # Keep only top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_width]
            
            # Move completed sequences
            new_beams = []
            for seq, score in beams:
                if seq[-1] == end_token:
                    completed_sequences.append((seq, score))
                else:
                    new_beams.append((seq, score))
            beams = new_beams
        
        # Add remaining beams to completed sequences
        completed_sequences.extend(beams)
        
        # Sort by score and return
        completed_sequences.sort(key=lambda x: x[1], reverse=True)
        return completed_sequences
```

# Optimization

## SGD
```python
class BaseOptimizer:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr
        self.state = {}
        
    def zero_grad(self):
        """Clear gradients for all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """Perform optimization step - to be implemented by subclasses"""
        raise NotImplementedError

class SGD(BaseOptimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr)
        
    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            param.data -= self.lr * param.grad.data
```

## Momentum
```python
class Momentum(BaseOptimizer):
    """SGD with Momentum using EWMA"""
    
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            # Initialize velocity if not exists
            if i not in self.state:
                self.state[i] = {'velocity': torch.zeros_like(param.data)}
            
            velocity = self.state[i]['velocity']
            
            # EWMA: v_t = β * v_{t-1} + (1-β) * g_t
            # But in practice, we use: v_t = β * v_{t-1} + g_t
            velocity.mul_(self.momentum).add_(param.grad.data)
            
            # Update parameters: θ = θ - lr * v_t
            param.data -= self.lr * velocity
```

## Nesterov
```python
class NesterovMomentum(BaseOptimizer):
    """Nesterov Accelerated Gradient"""
    
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            if i not in self.state:
                self.state[i] = {'velocity': torch.zeros_like(param.data)}
            
            velocity = self.state[i]['velocity']
            
            # Update velocity: v_t = β * v_{t-1} + g_t
            velocity.mul_(self.momentum).add_(param.grad.data)
            
            # Nesterov update: θ = θ - lr * (β * v_t + g_t)
            param.data -= self.lr * (self.momentum * velocity + param.grad.data)
```

## Adagrad
```python
class Adagrad(BaseOptimizer):
    """Adaptive Gradient Algorithm"""
    
    def __init__(self, params, lr=1e-2, eps=1e-8):
        super().__init__(params, lr)
        self.eps = eps
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            if i not in self.state:
                self.state[i] = {'sum_sq_grad': torch.zeros_like(param.data)}
            
            sum_sq_grad = self.state[i]['sum_sq_grad']
            
            # Accumulate squared gradients: G_t = G_{t-1} + g_t²
            sum_sq_grad.add_(param.grad.data.pow(2))
            
            # Update: θ = θ - lr * g_t / (√G_t + ε)
            param.data -= self.lr * param.grad.data / (sum_sq_grad.sqrt() + self.eps)
```

## RMSprop
```python
class RMSprop(BaseOptimizer):
    """RMSprop optimizer using EWMA for squared gradients"""
    
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha = alpha  # decay rate for EWMA
        self.eps = eps
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            if i not in self.state:
                self.state[i] = {'sq_avg': torch.zeros_like(param.data)}
            
            sq_avg = self.state[i]['sq_avg']
            
            # EWMA of squared gradients: v_t = α * v_{t-1} + (1-α) * g_t²
            sq_avg.mul_(self.alpha).addcmul_(param.grad.data, param.grad.data, value=1-self.alpha)
            
            # Update: θ = θ - lr * g_t / (√v_t + ε)
            param.data -= self.lr * param.grad.data / (sq_avg.sqrt() + self.eps)
```

## Adadelta
```python
class Adadelta(BaseOptimizer):
    """Adadelta optimizer - extension of Adagrad"""
    
    def __init__(self, params, rho=0.9, eps=1e-6):
        super().__init__(params, lr=1.0)  # lr not used in Adadelta
        self.rho = rho
        self.eps = eps
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            if i not in self.state:
                self.state[i] = {
                    'sq_avg': torch.zeros_like(param.data),
                    'acc_delta': torch.zeros_like(param.data)
                }
            
            sq_avg = self.state[i]['sq_avg']
            acc_delta = self.state[i]['acc_delta']
            
            # EWMA of squared gradients: v_t = ρ * v_{t-1} + (1-ρ) * g_t²
            sq_avg.mul_(self.rho).addcmul_(param.grad.data, param.grad.data, value=1-self.rho)
            
            # Compute update: Δθ_t = -√(u_{t-1} + ε) / √(v_t + ε) * g_t
            std = sq_avg.sqrt().add_(self.eps)
            delta = acc_delta.sqrt().add_(self.eps).div_(std).mul_(param.grad.data)
            
            # Update parameters
            param.data -= delta
            
            # EWMA of squared updates: u_t = ρ * u_{t-1} + (1-ρ) * Δθ_t²
            acc_delta.mul_(self.rho).addcmul_(delta, delta, value=1-self.rho)
```

## Adam
```python
class Adam(BaseOptimizer):
    """Adam optimizer - combines Momentum and RMSprop"""
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            if i not in self.state:
                self.state[i] = {
                    'm': torch.zeros_like(param.data),  # first moment
                    'v': torch.zeros_like(param.data)   # second moment
                }
            
            m = self.state[i]['m']
            v = self.state[i]['v']
            
            # EWMA of gradients: m_t = β₁ * m_{t-1} + (1-β₁) * g_t
            m.mul_(self.beta1).add_(param.grad.data, alpha=1-self.beta1)
            
            # EWMA of squared gradients: v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
            v.mul_(self.beta2).addcmul_(param.grad.data, param.grad.data, value=1-self.beta2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            
            # Update: θ = θ - lr * m̂_t / (√v̂_t + ε)
            param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
```

## AdamW
```python
class AdamW(BaseOptimizer):
    """AdamW optimizer - Adam with decoupled weight decay"""
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            if i not in self.state:
                self.state[i] = {
                    'm': torch.zeros_like(param.data),
                    'v': torch.zeros_like(param.data)
                }
            
            m = self.state[i]['m']
            v = self.state[i]['v']
            
            # Weight decay (decoupled from gradient)
            param.data.mul_(1 - self.lr * self.weight_decay)
            
            # EWMA of gradients and squared gradients
            m.mul_(self.beta1).add_(param.grad.data, alpha=1-self.beta1)
            v.mul_(self.beta2).addcmul_(param.grad.data, param.grad.data, value=1-self.beta2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            
            # Update parameters
            param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
```