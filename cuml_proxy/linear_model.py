from .proxy import ProxyEstimator

class LinearRegression(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("LinearRegression", **kwargs)

class LogisticRegression(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("LogisticRegression", **kwargs)

class Ridge(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("Ridge", **kwargs)

class Lasso(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("Lasso", **kwargs)

class ElasticNet(ProxyEstimator):
    def __init__(self, **kwargs): super().__init__("ElasticNet", **kwargs)
