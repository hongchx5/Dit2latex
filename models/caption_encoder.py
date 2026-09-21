"""
Caption（latex 公式）编码器：token id 序列 → 序列特征 (B, L, hidden_dim)。

数据流：
  caption.txt 每行 "<文件名>\t<空格 split 的 latex token>"，
  dictionary.txt 每行一个 latex token（词表）。

vocab 约定：
  PAD  = 0（padding 占位，embedding 恒为 0）
  UNK  = 1（词表外 token）
  其余 token 按 dictionary 行序从 2 开始编号。

输出序列特征作为 DiT 中 caption Cross-Attention 的 K/V（context_dim = hidden_dim）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

PAD_ID = 0
UNK_ID = 1
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocab(dictionary_path: str) -> Tuple[Dict[str, int], List[str]]:
    """
    从 dictionary.txt 构建词表。

    Args:
        dictionary_path: 每行一个 latex token 的文件路径。

    Returns:
        (token2id, id2token)。PAD=0, UNK=1，其余按行序从 2 开始。
    """
    token2id: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    with open(dictionary_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token and token not in token2id:
                token2id[token] = len(token2id)
    id2token = [""] * len(token2id)
    for tok, idx in token2id.items():
        id2token[idx] = tok
    return token2id, id2token


def encode_caption_string(formula: str, token2id: Dict[str, int]) -> List[int]:
    """
    将空格 split 的 latex token 字符串转为 id 序列（UNK 兜底）。

    Args:
        formula: 空格分隔的 token 字符串。
        token2id: 词表映射。

    Returns:
        token id 列表。
    """
    return [token2id.get(tok, UNK_ID) for tok in formula.split()]


class CaptionEncoder(nn.Module):
    """
    latex caption 编码器：Embedding + 可学习位置编码 + TransformerEncoder。

    输入 (B, L) token id + (B, L) padding mask（True=padding），
    输出 (B, L, hidden_dim) 序列特征。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        max_len: int = 256,
        num_layers: int = 1,
        num_heads: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # padding_idx=0 使 PAD token 的 embedding 恒为 0 且不参与梯度
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids:    (B, L) long，token id。
            padding_mask: (B, L) bool，True 表示 padding 位置（忽略）。

        Returns:
            (B, L, hidden_dim) 序列特征。
        """
        B, L = token_ids.shape
        assert L <= self.max_len, f"caption length {L} exceeds max_len {self.max_len}"

        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_embed(token_ids) + self.pos_embed(positions)  # (B, L, D)

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x
