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

## Penalty
```python
class RepetitionPenaltyController:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.reset()
    
    def reset(self):
        """Reset all penalty tracking."""
        self.token_counts = {}  # Track frequency of each token
        self.token_presence = set()  # Track which tokens have appeared
    
    def update(self, token_id: int):
        """Update tracking with a new token."""
        self.token_counts[token_id] = self.token_counts.get(token_id, 0) + 1
        self.token_presence.add(token_id)
    
    def apply_penalties(self, logits: torch.Tensor, 
                       frequency_penalty: float = 0.0,
                       presence_penalty: float = 0.0,
                       repetition_penalty: float = 1.0) -> torch.Tensor:
        """
        Apply various repetition penalties to logits.
        
        Args:
            logits: Raw model outputs [vocab_size]
            frequency_penalty: Penalty based on token frequency (higher = more penalty)
            presence_penalty: Penalty for tokens that have appeared before
            repetition_penalty: Classic repetition penalty (>1 = penalize, <1 = encourage)
        """
        modified_logits = logits.clone()
        
        # Apply frequency penalty
        if frequency_penalty != 0.0:
            for token_id, count in self.token_counts.items():
                modified_logits[token_id] -= frequency_penalty * count
        
        # Apply presence penalty
        if presence_penalty != 0.0:
            for token_id in self.token_presence:
                modified_logits[token_id] -= presence_penalty
        
        # Apply repetition penalty (classic approach)
        if repetition_penalty != 1.0:
            for token_id in self.token_presence:
                if modified_logits[token_id] > 0:
                    modified_logits[token_id] /= repetition_penalty
                else:
                    modified_logits[token_id] *= repetition_penalty
        
        return modified_logits
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

# Parallelism
```python
import torch.distributed as dist

class DDP:
    """
    Data Parallel: Each GPU has a full copy of the model.
    Gradients are synchronized across all GPUs after backward pass.
    """
    def __init__(self, model, rank, world_size):
        self.model = copy.deepcopy(model)
        self.rank = rank
        self.world_size = world_size
        
    def forward(self, x):
        # Each GPU processes its own batch
        return self.model(x)
    
    def backward_and_sync(self, loss):
        # Compute gradients locally
        loss.backward()
        
        # Synchronize gradients across all GPUs
        for param in self.model.parameters():
            if param.grad is not None:
                # All-reduce: sum gradients from all GPUs
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                # Average the gradients
                param.grad.data /= self.world_size
    
    def step(self, optimizer):
        # Each GPU updates its model with synchronized gradients
        optimizer.step()
```

## FSDP
```python
class FSDP:
    """
    Fully Sharded: Each GPU only stores a shard of model parameters.
    Parameters are gathered when needed and released after use.
    """
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        
        # Shard parameters across GPUs
        self.param_shards = {}
        self.shard_params()
    
    def shard_params(self):
        """Distribute parameters across GPUs"""
        for name, param in self.model.named_parameters():
            # Split parameter into shards
            shard_size = param.numel() // self.world_size
            start_idx = self.rank * shard_size
            end_idx = start_idx + shard_size if self.rank < self.world_size - 1 else param.numel()
            
            # Each GPU only keeps its shard
            self.param_shards[name] = {
                'local_shard': param.data.flatten()[start_idx:end_idx].clone(),
                'shape': param.shape,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
            # Clear full parameter to save memory
            param.data = torch.zeros_like(param.data)
    
    def gather_params(self, layer_name):
        """Gather all shards to reconstruct full parameter"""
        shard_info = self.param_shards[layer_name]
        full_param = torch.zeros(shard_info['shape'].numel())
        
        # All-gather: collect shards from all GPUs
        all_shards = [torch.zeros_like(shard_info['local_shard']) 
                      for _ in range(self.world_size)]
        dist.all_gather(all_shards, shard_info['local_shard'])
        
        # Reconstruct full parameter
        for rank, shard in enumerate(all_shards):
            shard_size = len(shard)
            start = rank * shard_size
            end = start + shard_size
            full_param[start:end] = shard
        
        return full_param.reshape(shard_info['shape'])
    
    def forward(self, x, layer):
        """Forward pass with parameter gathering"""
        # Gather parameters for this layer
        for name, param in layer.named_parameters():
            param.data = self.gather_params(name)
        
        # Compute forward pass
        output = layer(x)
        
        # Release parameters after use (zero them to save memory)
        for param in layer.parameters():
            param.data = torch.zeros_like(param.data)
        
        return output
    
    def reduce_scatter_gradients(self):
        """Each GPU only keeps gradients for its parameter shard"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                shard_info = self.param_shards[name]
                
                # Reduce-scatter: sum gradients and distribute shards
                grad_shard = torch.zeros_like(shard_info['local_shard'])
                dist.reduce_scatter(
                    grad_shard,
                    list(param.grad.flatten()[shard_info['start_idx']:shard_info['end_idx']])
                )
                
                # Store gradient shard
                self.param_shards[name]['grad_shard'] = grad_shard
```

## ZeRO
```python
class SimpleZeRO:
    """
    ZeRO: Partition optimizer states, gradients, and parameters across GPUs.
    Three stages of memory optimization.
    """
    def __init__(self, model, optimizer, rank, world_size, stage=1):
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.stage = stage
        
        # Partition assignments
        self.param_to_rank = {}
        self.assign_partitions()
    
    def assign_partitions(self):
        """Assign each parameter to a GPU"""
        params = list(self.model.parameters())
        for i, param in enumerate(params):
            assigned_rank = i % self.world_size
            self.param_to_rank[id(param)] = assigned_rank
    
    def zero_stage_1(self):
        """Stage 1: Partition optimizer states only"""
        # Each GPU only maintains optimizer states for its assigned parameters
        for param in self.model.parameters():
            if self.param_to_rank[id(param)] != self.rank:
                # Don't maintain optimizer state for this parameter
                self.optimizer.state[param] = {}
    
    def zero_stage_2(self):
        """Stage 2: Partition optimizer states + gradients"""
        self.zero_stage_1()
        
        # After backward, each GPU only keeps gradients for its parameters
        for param in self.model.parameters():
            if param.grad is not None:
                if self.param_to_rank[id(param)] == self.rank:
                    # This GPU owns this parameter's gradient
                    dist.reduce(param.grad, dst=self.param_to_rank[id(param)])
                else:
                    # Clear gradient to save memory
                    param.grad = None
    
    def zero_stage_3(self):
        """Stage 3: Partition everything (params + gradients + optimizer states)"""
        self.zero_stage_2()
        
        # Each GPU only keeps parameters it owns
        for param in self.model.parameters():
            if self.param_to_rank[id(param)] != self.rank:
                # Replace with empty tensor to save memory
                param.data = torch.empty(0)
    
    def gather_params_for_forward(self):
        """Gather parameters when needed (Stage 3)"""
        if self.stage < 3:
            return
        
        for param in self.model.parameters():
            if param.data.numel() == 0:
                # Need to gather this parameter from owner
                owner_rank = self.param_to_rank[id(param)]
                dist.broadcast(param.data, src=owner_rank)
    
    def step(self):
        """Optimizer step with ZeRO optimization"""
        if self.stage == 1:
            self.zero_stage_1()
        elif self.stage == 2:
            self.zero_stage_2()
        elif self.stage == 3:
            self.zero_stage_3()
        
        # Each GPU updates only its assigned parameters
        for param in self.model.parameters():
            if self.param_to_rank[id(param)] == self.rank and param.grad is not None:
                # Update parameter
                self.optimizer.step()
                
                # Broadcast updated parameter to all GPUs (for stage 3)
                if self.stage == 3:
                    dist.broadcast(param.data, src=self.rank)
```

# Tokenization

## BPE
```python
class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = []  # List of merge operations in order
        self.vocab = {}  # Final vocabulary
        
    def get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Get frequency of each word in the corpus"""
        word_freq = Counter()
        for text in texts:
            # Simple whitespace tokenization
            words = text.lower().split()
            for word in words:
                # Add end-of-word token
                word_freq[word + '</w>'] += 1
        return dict(word_freq)
    
    def get_character_vocab(self, word_freq: Dict[str, int]) -> Set[str]:
        """Get initial vocabulary of all characters"""
        vocab = set()
        for word in word_freq:
            for char in word:
                vocab.add(char)
        return vocab
    
    def get_pairs(self, word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get all adjacent pairs and their frequencies"""
        pairs = defaultdict(int)
        
        for word, freq in word_freq.items():
            # Split word into characters
            symbols = word.split() if ' ' in word else list(word)
            
            # Count adjacent pairs
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
                
        return dict(pairs)
    
    def merge_vocab(self, pair: Tuple[str, str], word_freq: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in vocabulary"""
        new_word_freq = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freq.items():
            # Replace the pair with merged version
            new_word = word.replace(bigram, replacement)
            new_word_freq[new_word] = freq
            
        return new_word_freq
    
    def train(self, texts: List[str]):
        """Train BPE on a corpus of texts"""
        print("Training BPE tokenizer...")
        
        # Get word frequencies
        word_freq = self.get_word_frequencies(texts)
        
        # Initialize vocabulary with characters
        vocab = self.get_character_vocab(word_freq)
        
        # Add spaces between characters for merging
        spaced_word_freq = {}
        for word, freq in word_freq.items():
            spaced_word = ' '.join(word)
            spaced_word_freq[spaced_word] = freq
        
        word_freq = spaced_word_freq
        
        # Perform merges until we reach vocab_size
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            pairs = self.get_pairs(word_freq)
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            word_freq = self.merge_vocab(best_pair, word_freq)
            self.merges.append(best_pair)
            vocab.add(''.join(best_pair))
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} merges")
        
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[str]:
        """Encode text using trained BPE"""
        if not self.vocab:
            raise ValueError("Tokenizer not trained yet!")
        
        words = text.lower().split()
        tokens = []
        
        for word in words:
            word = word + '</w>'
            word_tokens = list(word)
            
            # Apply merges in order
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        # Merge the pair
                        merged = word_tokens[i] + word_tokens[i + 1]
                        word_tokens = word_tokens[:i] + [merged] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens
```

## WordPiece
```python
class WordPieceTokenizer:
    def __init__(self, vocab_size: int = 1000, unk_token: str = '[UNK]'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {}
        
    def get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Get frequency of each word in the corpus"""
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] += 1
        return dict(word_freq)
    
    def get_initial_vocab(self, word_freq: Dict[str, int]) -> Set[str]:
        """Get initial vocabulary of characters"""
        vocab = set([self.unk_token])
        for word in word_freq:
            for char in word:
                vocab.add(char)
        return vocab
    
    def get_subword_candidates(self, word_freq: Dict[str, int], vocab: Set[str]) -> Dict[str, float]:
        """Get all possible subword candidates and their scores"""
        candidates = {}
        
        for word, freq in word_freq.items():
            # Generate all possible subwords
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    subword = word[i:j]
                    if i > 0:
                        subword = '##' + subword  # WordPiece convention
                    
                    if subword not in vocab and len(subword) > 1:
                        if subword not in candidates:
                            candidates[subword] = 0
                        candidates[subword] += freq
        
        return candidates
    
    def train(self, texts: List[str]):
        """Train WordPiece tokenizer"""
        print("Training WordPiece tokenizer...")
        
        word_freq = self.get_word_frequencies(texts)
        vocab = self.get_initial_vocab(word_freq)
        
        while len(vocab) < self.vocab_size:
            candidates = self.get_subword_candidates(word_freq, vocab)
            
            if not candidates:
                break
            
            # Select best candidate (highest frequency)
            best_candidate = max(candidates, key=candidates.get)
            vocab.add(best_candidate)
            
            if len(vocab) % 100 == 0:
                print(f"Vocabulary size: {len(vocab)}")
        
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        print(f"Training complete. Final vocabulary size: {len(self.vocab)}")
    
    def encode_word(self, word: str) -> List[str]:
        """Encode a single word using greedy longest-match"""
        if not self.vocab:
            raise ValueError("Tokenizer not trained yet!")
        
        tokens = []
        i = 0
        
        while i < len(word):
            longest_match = None
            longest_length = 0
            
            # Find longest matching subword
            for j in range(i + 1, len(word) + 1):
                subword = word[i:j]
                if i > 0:
                    subword = '##' + subword
                
                if subword in self.vocab and len(subword) > longest_length:
                    longest_match = subword
                    longest_length = len(subword)
            
            if longest_match:
                tokens.append(longest_match)
                i += longest_length if not longest_match.startswith('##') else longest_length - 2
            else:
                tokens.append(self.unk_token)
                i += 1
        
        return tokens
    
    def encode(self, text: str) -> List[str]:
        """Encode text using WordPiece"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            tokens.extend(self.encode_word(word))
        
        return tokens
```

## Unigram
```python
class UnigramTokenizer:
    def __init__(self, vocab_size: int = 1000, unk_token: str = '<UNK>'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {}
        self.token_probs = {}
        
    def get_initial_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Get initial large vocabulary with all possible substrings"""
        substring_freq = Counter()
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Add all possible substrings
                for i in range(len(word)):
                    for j in range(i + 1, len(word) + 1):
                        substring = word[i:j]
                        substring_freq[substring] += 1
        
        return dict(substring_freq)
    
    def viterbi_segment(self, word: str, vocab_probs: Dict[str, float]) -> List[str]:
        """Find best segmentation using Viterbi algorithm"""
        n = len(word)
        if n == 0:
            return []
        
        # Dynamic programming arrays
        best_score = [-float('inf')] * (n + 1)
        best_score[0] = 0
        previous = [None] * (n + 1)
        
        for i in range(1, n + 1):
            for j in range(i):
                token = word[j:i]
                if token in vocab_probs:
                    score = best_score[j] + math.log(vocab_probs[token])
                    if score > best_score[i]:
                        best_score[i] = score
                        previous[i] = j
        
        # Backtrack to get segmentation
        tokens = []
        i = n
        while i > 0:
            j = previous[i]
            if j is not None:
                tokens.append(word[j:i])
                i = j
            else:
                tokens.append(self.unk_token)
                break
        
        return list(reversed(tokens))
    
    def calculate_loss(self, texts: List[str], vocab_probs: Dict[str, float]) -> float:
        """Calculate likelihood loss for current vocabulary"""
        total_loss = 0
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                segmentation = self.viterbi_segment(word, vocab_probs)
                word_loss = sum(-math.log(vocab_probs.get(token, 1e-10)) for token in segmentation)
                total_loss += word_loss
        
        return total_loss
    
    def train(self, texts: List[str]):
        """Train Unigram tokenizer"""
        print("Training Unigram tokenizer...")
        
        # Start with large vocabulary
        initial_vocab = self.get_initial_vocab(texts)
        
        # Keep only most frequent substrings (reduce computational cost)
        sorted_vocab = sorted(initial_vocab.items(), key=lambda x: x[1], reverse=True)
        current_vocab = dict(sorted_vocab[:self.vocab_size * 3])  # Start with 3x target size
        
        # Add single characters to ensure coverage
        all_chars = set()
        for text in texts:
            all_chars.update(text.lower())
        for char in all_chars:
            if char not in current_vocab:
                current_vocab[char] = 1
        
        current_vocab[self.unk_token] = 1
        
        # Iteratively reduce vocabulary
        while len(current_vocab) > self.vocab_size:
            # Calculate token probabilities
            total_freq = sum(current_vocab.values())
            vocab_probs = {token: freq / total_freq for token, freq in current_vocab.items()}
            
            # Find token that contributes least to likelihood
            best_token_to_remove = None
            best_loss_increase = float('inf')
            
            for token in list(current_vocab.keys()):
                if token == self.unk_token or len(token) == 1:  # Don't remove UNK or single chars
                    continue
                
                # Temporarily remove token
                temp_vocab = current_vocab.copy()
                del temp_vocab[token]
                
                # Recalculate probabilities
                temp_total = sum(temp_vocab.values())
                temp_probs = {t: f / temp_total for t, f in temp_vocab.items()}
                
                # Calculate loss increase (simplified)
                loss_increase = current_vocab[token]  # Approximate
                
                if loss_increase < best_loss_increase:
                    best_loss_increase = loss_increase
                    best_token_to_remove = token
            
            if best_token_to_remove:
                del current_vocab[best_token_to_remove]
            
            if len(current_vocab) % 100 == 0:
                print(f"Vocabulary size: {len(current_vocab)}")
        
        # Final vocabulary and probabilities
        total_freq = sum(current_vocab.values())
        self.vocab = {token: i for i, token in enumerate(sorted(current_vocab.keys()))}
        self.token_probs = {token: freq / total_freq for token, freq in current_vocab.items()}
        
        print(f"Training complete. Final vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[str]:
        """Encode text using Unigram tokenizer"""
        if not self.vocab:
            raise ValueError("Tokenizer not trained yet!")
        
        words = text.lower().split()
        tokens = []
        
        for word in words:
            word_tokens = self.viterbi_segment(word, self.token_probs)
            tokens.extend(word_tokens)
        
        return tokens
```

# Inference
## KV Cache
```python
class KVCache:
    """Key-Value cache for transformer attention layers."""
    
    def __init__(self, max_seq_len: int, n_heads: int, head_dim: int, n_layers: int):
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.reset()
    
    def reset(self):
        """Reset the cache."""
        self.cache = {}
        self.seq_len = 0
    
    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached K,V for a layer."""
        if layer_idx not in self.cache:
            return None, None
        return self.cache[layer_idx]
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Update cache with new K,V tensors."""
        if layer_idx not in self.cache:
            # Initialize cache for this layer
            self.cache[layer_idx] = (k, v)
        else:
            # Concatenate with existing cache
            cached_k, cached_v = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([cached_k, k], dim=-2),  # Concat along sequence dimension
                torch.cat([cached_v, v], dim=-2)
            )
        
        # Update sequence length
        self.seq_len = self.cache[layer_idx][0].shape[-2]


class MultiHeadAttentionWithKVCache(nn.Module):
    """Multi-head attention with KV caching support."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, 
                layer_idx: int = 0) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Handle KV caching
        if kv_cache is not None:
            cached_k, cached_v = kv_cache.get(layer_idx)
            if cached_k is not None:
                # Use cached K,V and only compute for new tokens
                k = torch.cat([cached_k, k], dim=1)
                v = torch.cat([cached_v, v], dim=1)
            
            # Update cache
            kv_cache.update(layer_idx, k[:, -seq_len:], v[:, -seq_len:])
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask (only look at previous tokens)
        if scores.size(-1) > 1:
            mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(output)
```