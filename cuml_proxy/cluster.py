from .proxy import ProxyEstimator

class KMeans(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("KMeans", **kwargs)

class DBSCAN(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("DBSCAN", **kwargs)
