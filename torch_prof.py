import torch
import torch.nn as nn
import torch.profiler

# A simplified Transformer Encoder Layer to demonstrate profiling
class SimpleTransformerLayer(nn.Module):
    """
    A basic implementation of a Transformer Encoder Layer.
    It includes Multi-Head Self-Attention, Layer Normalization, and a Feed-Forward Network (MLP).
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-Forward Network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass for the Transformer Layer.
        We will wrap key sections with record_function to create distinct profiler labels.
        """
        # 1. Self-Attention Block
        with torch.profiler.record_function("Self-Attention"):
            # The multi-head attention layer expects query, key, and value inputs.
            # For self-attention, the source tensor is used for all three.
            attn_output, _ = self.self_attention(src, src, src)

        # Residual connection with dropout and LayerNorm
        with torch.profiler.record_function("Add & Norm 1"):
            src = src + self.dropout1(attn_output)
            src = self.norm1(src)

        # 2. MLP (Feed-Forward) Block
        with torch.profiler.record_function("MLP"):
            mlp_output = self.mlp(src)

        # Residual connection with dropout and LayerNorm
        with torch.profiler.record_function("Add & Norm 2"):
            src = src + self.dropout2(mlp_output)
            src = self.norm2(src)

        return src

def main():
    # --- Model and Input Configuration ---
    batch_size = 16
    seq_length = 128
    embed_dim = 512  # d_model
    num_heads = 8
    ff_dim = 2048   # Hidden dimension in MLP

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate Model and Create Dummy Input ---
    model = SimpleTransformerLayer(embed_dim, num_heads, ff_dim).to(device)
    model.eval() # Set to evaluation mode for profiling inference

    # Create a random input tensor
    dummy_input = torch.randn(batch_size, seq_length, embed_dim).to(device)

    print("\nModel and input tensor created. Starting profiler...")
    print("-" * 50)

    # --- Run the Profiler ---
    # The profiler context manager will trace the execution and performance.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA, # Only include if CUDA is available
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/transformer'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # The profiler will step through a schedule of wait, warmup, and active phases.
        # We need to call `prof.step()` at the end of each iteration.
        for i in range(10):
            # We only care about the forward pass for this example
            with torch.no_grad():
                model(dummy_input)
            prof.step() # Notify the profiler that a step is complete

    # --- Print Profiler Results ---
    print("Profiler run complete. Printing summary...")
    print("-" * 50)

    # Print a summary of the results to the console, grouped by our custom labels.
    # The `group_by_input_shape` is useful for seeing how different tensor sizes perform.
    # The `group_by_stack_n` helps attribute time to specific lines of code.
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))

    print("\n" + "-" * 50)
    print("To view the detailed trace, run the following command in your terminal:")
    print("tensorboard --logdir=./log")
    print("-" * 50)


if __name__ == "__main__":
    main()

