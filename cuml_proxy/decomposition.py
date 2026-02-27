from .proxy import ProxyEstimator

class PCA(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("PCA", **kwargs)

class TruncatedSVD(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("TruncatedSVD", **kwargs)
