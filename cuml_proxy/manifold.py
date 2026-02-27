from .proxy import ProxyEstimator

class TSNE(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("TSNE", **kwargs)

class UMAP(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("UMAP", **kwargs)
