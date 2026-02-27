from .proxy import ProxyEstimator

class SVC(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("SVC", **kwargs)

class SVR(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("SVR", **kwargs)
