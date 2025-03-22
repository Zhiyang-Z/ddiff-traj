import torch

def gen_attn_mask(size: int) -> torch.tensor:
    """generate attention mask with shape: (size, size)"""
    mask = torch.tril(torch.ones(size, size))
    return torch.where(mask == 0, float('-inf'), float(0.0))

def gen_padding_mask(x: torch.tensor, padding_token: int) -> torch.tensor:
    """generate padding mask where padding position filled with True, otherwise, False."""
    return x == padding_token

def gen_pos_encoding(max_len:int, dim: int) -> torch.tensor:
    if dim % 2: raise ValueError('only even dim is implemented.')
    pos, a, exponent = torch.arange(max_len), torch.full((dim//2,), 10000), torch.arange(0, dim, 2)/dim
    denominator = torch.reciprocal(torch.pow(a, exponent))
    degree = pos.unsqueeze(1).mul(denominator.unsqueeze(0))
    # fill positional encoding.
    pos_encoding = torch.zeros(max_len, dim)
    pos_encoding[:,::2], pos_encoding[:,1::2] = torch.sin(degree), torch.cos(degree)
    return pos_encoding

# test
if __name__ == '__main__':
    print(gen_attn_mask(7))
    print(gen_padding_mask(torch.tril(torch.ones(7,7)), 0))
    import matplotlib.pyplot as plt
    pos_encoding = gen_pos_encoding(100, 768)
    cax = plt.matshow(pos_encoding)
    plt.gcf().colorbar(cax)
    plt.show()