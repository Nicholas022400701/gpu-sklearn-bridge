"""
cuml_proxy – Windows-side sklearn-compatible wrapper.
Transparently forwards all calls to the GPU bridge server running in WSL2.

Usage:
    from cuml_proxy.linear_model import LinearRegression
    from cuml_proxy.cluster import KMeans
    # ... same API as sklearn / cuML
"""
from .proxy import ProxyEstimator, _BRIDGE_URL, wait_for_server  # noqa: F401
