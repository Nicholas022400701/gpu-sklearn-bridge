from .proxy import ProxyEstimator

class RandomForestClassifier(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("RandomForestClassifier", **kwargs)

class RandomForestRegressor(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("RandomForestRegressor", **kwargs)
