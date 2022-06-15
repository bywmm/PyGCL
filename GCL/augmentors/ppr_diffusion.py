from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import compute_ppr, compute_ppr_inductive


class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        if isinstance(edge_index, list):
            for i, (e_index, _, size) in enumerate(edge_index):
                e_index, _ = compute_ppr_inductive(
                    e_index, edge_weights[i] if edge_weights is not None else None, size[0], 
                    alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
                )
                edge_index[i] = (e_index, _, size)
        else:
            edge_index, edge_weights = compute_ppr(
                edge_index, edge_weights,
                alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
            )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res
