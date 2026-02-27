from .proxy import ProxyEstimator

class KNeighborsClassifier(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("KNeighborsClassifier", **kwargs)

class KNeighborsRegressor(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("KNeighborsRegressor", **kwargs)

class NearestNeighbors(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("NearestNeighbors", **kwargs)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """返回 (distances, indices) tuple，或仅 indices（return_distance=False）。"""
        kw = {"return_distance": return_distance}
        if n_neighbors is not None:
            kw["n_neighbors"] = n_neighbors
        return self._call("kneighbors", X, **kw)
