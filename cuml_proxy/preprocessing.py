from .proxy import ProxyEstimator

class StandardScaler(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("StandardScaler", **kwargs)

class MinMaxScaler(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("MinMaxScaler", **kwargs)

class LabelEncoder(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("LabelEncoder", **kwargs)
