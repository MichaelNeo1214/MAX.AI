import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        block_size
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V dibuat sekaligus
        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3
        )

        # Output projection
        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        # Causal mask
        mask = torch.tril(
            torch.ones(
                block_size,
                block_size
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                block_size,
                block_size
            )
        )


    def forward(self, x):

        batch_size, seq_len, embed_dim = x.shape

        # -----------------------------------------------------
        # QKV
        # -----------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = torch.chunk(
            qkv,
            3,
            dim=-1
        )

        # -----------------------------------------------------
        # Pisahkan menjadi attention heads
        # -----------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # -----------------------------------------------------
        # Attention Score
        # -----------------------------------------------------

        scores = (
            q @ k.transpose(-2, -1)
        ) / (
            self.head_dim ** 0.5
        )

        # -----------------------------------------------------
        # Causal Mask
        # -----------------------------------------------------

        scores = scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf")
        )

        # -----------------------------------------------------
        # Softmax
        # -----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -----------------------------------------------------
        # Weighted Value
        # -----------------------------------------------------

        attention_output = (
            attention_weights @ v
        )

        # -----------------------------------------------------
        # Gabungkan heads
        # -----------------------------------------------------

        attention_output = attention_output.transpose(
            1,
            2
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            seq_len,
            embed_dim
        )

        # -----------------------------------------------------
        # Output projection
        # -----------------------------------------------------

        output = self.out_proj(
            attention_output
        )

        return output


class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim,
        hidden_dim
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embed_dim
            )
        )


    def forward(self, x):

        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_dim,
        block_size
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size
        )

        self.layer_norm_2 = nn.LayerNorm(
            embed_dim
        )

        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )


    def forward(self, x):

        # Residual connection + attention
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Residual connection + feed forward
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


class MichaelAI(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        block_size=32
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size

        # -----------------------------------------------------
        # Token Embedding
        # -----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        # -----------------------------------------------------
        # Positional Embedding
        # -----------------------------------------------------

        self.position_embedding = nn.Embedding(
            block_size,
            embed_dim
        )

        # -----------------------------------------------------
        # Transformer Blocks
        # -----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    block_size=block_size
                )
                for _ in range(num_layers)
            ]
        )

        # -----------------------------------------------------
        # Final LayerNorm
        # -----------------------------------------------------

        self.final_norm = nn.LayerNorm(
            embed_dim
        )

        # -----------------------------------------------------
        # Output Head
        # -----------------------------------------------------

        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size
        )


    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        # Pastikan sequence tidak melebihi context
        if seq_len > self.block_size:

            raise ValueError(
                "Sequence length melebihi block_size"
            )

        # -----------------------------------------------------
        # Token embedding
        # -----------------------------------------------------

        token_embeddings = self.token_embedding(
            input_ids
        )

        # -----------------------------------------------------
        # Position embedding
        # -----------------------------------------------------

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # -----------------------------------------------------
        # Gabungkan
        # -----------------------------------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        # -----------------------------------------------------
        # Transformer
        # -----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # -----------------------------------------------------
        # Final normalization
        # -----------------------------------------------------

        x = self.final_norm(x)

        # -----------------------------------------------------
        # Vocabulary logits
        # -----------------------------------------------------
        
        logits = self.lm_head(x)

        return logits