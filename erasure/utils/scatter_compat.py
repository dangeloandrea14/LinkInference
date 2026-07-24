import torch

try:                                     # PyG's implementation, handles broadcasting
    from torch_geometric.utils import scatter as _pyg_scatter
except ImportError:                      # pragma: no cover
    _pyg_scatter = None


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    """Drop-in replacement for `torch_scatter.scatter_add`.
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0

    if out is None and _pyg_scatter is not None:
        return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')

    # Pure-torch path: broadcast index across the non-scatter dimensions of src.
    idx = index
    if src.dim() > 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        idx = index.view(shape).expand_as(src)

    if out is None:
        size = list(src.shape)
        size[dim] = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    return out.scatter_add_(dim, idx, src)
