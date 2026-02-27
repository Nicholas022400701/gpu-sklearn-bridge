"""
Model parameter configurations for all supported cuML / sklearn models.
Each entry defines the task type and all configurable parameters with
type, default, constraints, and a description.
"""

MODEL_CONFIGS = {
    # ── Linear Models ──────────────────────────────────────────────────────────
    "LinearRegression": {
        "task": "regression",
        "module": "linear_model",
        "desc": "Ordinary Least Squares linear regression.",
        "params": {
            "fit_intercept":  {"type": "bool",  "default": True,   "desc": "Whether to calculate the intercept."},
            "normalize":      {"type": "bool",  "default": False,  "desc": "Normalize features before fitting (deprecated in newer versions)."},
            "algorithm":      {"type": "select","default": "eig",  "options": ["eig","svd","qr","svd-qr"], "desc": "Solver algorithm."},
        },
    },
    "Ridge": {
        "task": "regression",
        "module": "linear_model",
        "desc": "Ridge regression with L2 regularization.",
        "params": {
            "alpha":          {"type": "float", "default": 1.0,    "min": 0.0,  "desc": "Regularization strength."},
            "fit_intercept":  {"type": "bool",  "default": True,   "desc": "Fit intercept."},
            "normalize":      {"type": "bool",  "default": False,  "desc": "Normalize features before fitting."},
            "solver":         {"type": "select","default": "auto", "options": ["auto","svd","cholesky","lsqr","sparse_cg","sag","saga","eig","cd"], "desc": "Solver."},
        },
    },
    "Lasso": {
        "task": "regression",
        "module": "linear_model",
        "desc": "Lasso regression with L1 regularization.",
        "params": {
            "alpha":          {"type": "float", "default": 1.0,    "min": 0.0,  "desc": "Regularization strength."},
            "fit_intercept":  {"type": "bool",  "default": True,   "desc": "Fit intercept."},
            "normalize":      {"type": "bool",  "default": False,  "desc": "Normalize features before fitting."},
            "max_iter":       {"type": "int",   "default": 1000,   "min": 1,    "desc": "Max iterations."},
            "tol":            {"type": "float", "default": 1e-4,   "min": 0.0,  "desc": "Tolerance for optimization."},
            "selection":      {"type": "select","default": "cyclic","options": ["cyclic","random"], "desc": "Feature update order."},
        },
    },
    "ElasticNet": {
        "task": "regression",
        "module": "linear_model",
        "desc": "ElasticNet with combined L1 and L2 regularization.",
        "params": {
            "alpha":          {"type": "float", "default": 1.0,    "min": 0.0,   "desc": "Regularization strength."},
            "l1_ratio":       {"type": "float", "default": 0.5,    "min": 0.0, "max": 1.0, "desc": "L1 / (L1+L2) mix ratio."},
            "fit_intercept":  {"type": "bool",  "default": True,   "desc": "Fit intercept."},
            "normalize":      {"type": "bool",  "default": False,  "desc": "Normalize features."},
            "max_iter":       {"type": "int",   "default": 1000,   "min": 1,     "desc": "Max iterations."},
            "tol":            {"type": "float", "default": 1e-4,   "min": 0.0,   "desc": "Tolerance."},
            "selection":      {"type": "select","default": "cyclic","options": ["cyclic","random"], "desc": "Feature update order."},
        },
    },
    "LogisticRegression": {
        "task": "classification",
        "module": "linear_model",
        "desc": "Logistic regression classifier.",
        "params": {
            "penalty":        {"type": "select","default": "l2",   "options": ["none","l1","l2","elasticnet"], "desc": "Regularization type."},
            "tol":            {"type": "float", "default": 1e-4,   "min": 0.0,   "desc": "Stopping tolerance."},
            "C":              {"type": "float", "default": 1.0,    "min": 1e-6,  "desc": "Inverse regularization strength."},
            "fit_intercept":  {"type": "bool",  "default": True,   "desc": "Fit intercept."},
            "max_iter":       {"type": "int",   "default": 1000,   "min": 1,     "desc": "Max iterations."},
            "solver":         {"type": "select","default": "qn",   "options": ["qn","lbfgs","ols","newton-cg","sgd","cd"], "desc": "Solver."},
            "l1_ratio":       {"type": "float", "default": None,   "min": 0.0, "max": 1.0, "nullable": True, "desc": "L1 ratio for elasticnet."},
        },
    },
    "MBSGDClassifier": {
        "task": "classification",
        "module": "linear_model",
        "desc": "Mini-batch SGD classifier (supports partial_fit / iterative training).",
        "params": {
            "loss":           {"type": "select","default": "hinge","options": ["hinge","log","modified_huber","squared_hinge","perceptron","squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"], "desc": "Loss function."},
            "penalty":        {"type": "select","default": "l2",   "options": ["none","l1","l2","elasticnet"], "desc": "Regularization."},
            "alpha":          {"type": "float", "default": 1e-4,   "min": 0.0,   "desc": "Regularization strength."},
            "l1_ratio":       {"type": "float", "default": 0.15,   "min": 0.0, "max": 1.0, "desc": "L1 ratio for elasticnet."},
            "fit_intercept":  {"type": "bool",  "default": True,   "desc": "Fit intercept."},
            "epochs":         {"type": "int",   "default": 100,    "min": 1,     "desc": "Training epochs."},
            "tol":            {"type": "float", "default": 1e-3,   "min": 0.0,   "desc": "Stopping tolerance."},
            "shuffle":        {"type": "bool",  "default": True,   "desc": "Shuffle data each epoch."},
            "learning_rate":  {"type": "select","default": "adaptive","options": ["constant","optimal","invscaling","adaptive"], "desc": "Learning rate schedule."},
            "eta0":           {"type": "float", "default": 0.001,  "min": 0.0,   "desc": "Initial learning rate."},
            "batch_size":     {"type": "int",   "default": 512,    "min": 1,     "desc": "Mini-batch size."},
        },
    },
    "MBSGDRegressor": {
        "task": "regression",
        "module": "linear_model",
        "desc": "Mini-batch SGD regressor (supports partial_fit / iterative training).",
        "params": {
            "loss":           {"type": "select","default": "squared_loss","options": ["squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"], "desc": "Loss function."},
            "penalty":        {"type": "select","default": "l2",   "options": ["none","l1","l2","elasticnet"], "desc": "Regularization."},
            "alpha":          {"type": "float", "default": 1e-4,   "min": 0.0,   "desc": "Regularization strength."},
            "l1_ratio":       {"type": "float", "default": 0.15,   "min": 0.0, "max": 1.0, "desc": "L1 ratio for elasticnet."},
            "fit_intercept":  {"type": "bool",  "default": True,   "desc": "Fit intercept."},
            "epochs":         {"type": "int",   "default": 100,    "min": 1,     "desc": "Training epochs."},
            "tol":            {"type": "float", "default": 1e-3,   "min": 0.0,   "desc": "Stopping tolerance."},
            "shuffle":        {"type": "bool",  "default": True,   "desc": "Shuffle data each epoch."},
            "learning_rate":  {"type": "select","default": "adaptive","options": ["constant","optimal","invscaling","adaptive"], "desc": "Learning rate schedule."},
            "eta0":           {"type": "float", "default": 0.001,  "min": 0.0,   "desc": "Initial learning rate."},
            "batch_size":     {"type": "int",   "default": 512,    "min": 1,     "desc": "Mini-batch size."},
        },
    },
    # ── SVM ───────────────────────────────────────────────────────────────────
    "SVC": {
        "task": "classification",
        "module": "svm",
        "desc": "Support Vector Classification.",
        "params": {
            "kernel":         {"type": "select","default": "rbf",  "options": ["linear","poly","rbf","sigmoid"], "desc": "Kernel type."},
            "C":              {"type": "float", "default": 1.0,    "min": 1e-6,  "desc": "Regularization parameter."},
            "degree":         {"type": "int",   "default": 3,      "min": 1,     "desc": "Degree for poly kernel."},
            "gamma":          {"type": "select","default": "scale","options": ["scale","auto"], "desc": "Kernel coefficient."},
            "coef0":          {"type": "float", "default": 0.0,    "desc": "Independent term for poly/sigmoid kernel."},
            "tol":            {"type": "float", "default": 1e-3,   "min": 0.0,   "desc": "Stopping tolerance."},
            "max_iter":       {"type": "int",   "default": -1,     "desc": "Max iterations (-1 = no limit)."},
            "probability":    {"type": "bool",  "default": False,  "desc": "Enable probability estimates."},
        },
    },
    "SVR": {
        "task": "regression",
        "module": "svm",
        "desc": "Support Vector Regression.",
        "params": {
            "kernel":         {"type": "select","default": "rbf",  "options": ["linear","poly","rbf","sigmoid"], "desc": "Kernel type."},
            "C":              {"type": "float", "default": 1.0,    "min": 1e-6,  "desc": "Regularization parameter."},
            "degree":         {"type": "int",   "default": 3,      "min": 1,     "desc": "Degree for poly kernel."},
            "gamma":          {"type": "select","default": "scale","options": ["scale","auto"], "desc": "Kernel coefficient."},
            "coef0":          {"type": "float", "default": 0.0,    "desc": "Independent term."},
            "tol":            {"type": "float", "default": 1e-3,   "min": 0.0,   "desc": "Stopping tolerance."},
            "epsilon":        {"type": "float", "default": 0.1,    "min": 0.0,   "desc": "Epsilon in epsilon-SVR model."},
            "max_iter":       {"type": "int",   "default": -1,     "desc": "Max iterations (-1 = no limit)."},
        },
    },
    # ── Clustering ────────────────────────────────────────────────────────────
    "KMeans": {
        "task": "clustering",
        "module": "cluster",
        "desc": "K-Means clustering algorithm.",
        "params": {
            "n_clusters":     {"type": "int",   "default": 8,      "min": 1,     "desc": "Number of clusters."},
            "init":           {"type": "select","default": "scalable-k-means++","options": ["scalable-k-means++","k-means||","random"], "desc": "Centroid initialization method."},
            "max_iter":       {"type": "int",   "default": 300,    "min": 1,     "desc": "Max iterations."},
            "tol":            {"type": "float", "default": 1e-4,   "min": 0.0,   "desc": "Convergence tolerance."},
            "n_init":         {"type": "int",   "default": 1,      "min": 1,     "desc": "Number of random initializations."},
            "random_state":   {"type": "int",   "default": 42,     "desc": "Random seed."},
            "oversampling_factor": {"type": "float","default": 2.0,"min": 1.0,   "desc": "Oversampling factor for k-means||."},
        },
    },
    "DBSCAN": {
        "task": "clustering",
        "module": "cluster",
        "desc": "Density-Based Spatial Clustering of Applications with Noise.",
        "params": {
            "eps":            {"type": "float", "default": 0.5,    "min": 0.0,   "desc": "Max distance between neighbors."},
            "min_samples":    {"type": "int",   "default": 5,      "min": 1,     "desc": "Min samples in a neighborhood."},
            "metric":         {"type": "select","default": "euclidean","options": ["euclidean","cosine","l1","l2"], "desc": "Distance metric."},
            "algorithm":      {"type": "select","default": "brute","options": ["brute","rbc"], "desc": "Algorithm for neighbor search."},
            "max_mbytes_per_batch": {"type": "int","default": None,"nullable": True, "desc": "Max memory per batch (MB)."},
        },
    },
    # ── Decomposition ─────────────────────────────────────────────────────────
    "PCA": {
        "task": "decomposition",
        "module": "decomposition",
        "desc": "Principal Component Analysis.",
        "params": {
            "n_components":   {"type": "int",   "default": None,   "min": 1, "nullable": True, "desc": "Number of components (None = all)."},
            "svd_solver":     {"type": "select","default": "full", "options": ["full","jacobi","auto"], "desc": "SVD solver."},
            "whiten":         {"type": "bool",  "default": False,  "desc": "Whiten components."},
            "random_state":   {"type": "int",   "default": 42,     "desc": "Random seed."},
            "iterated_power": {"type": "int",   "default": 15,     "min": 1, "desc": "Power iterations for randomized SVD."},
        },
    },
    "TruncatedSVD": {
        "task": "decomposition",
        "module": "decomposition",
        "desc": "Truncated Singular Value Decomposition.",
        "params": {
            "n_components":   {"type": "int",   "default": 2,      "min": 1,     "desc": "Number of components."},
            "algorithm":      {"type": "select","default": "full", "options": ["full","jacobi"], "desc": "SVD algorithm."},
            "random_state":   {"type": "int",   "default": 42,     "desc": "Random seed."},
            "n_iter":         {"type": "int",   "default": 4,      "min": 1,     "desc": "Number of iterations for randomized SVD."},
        },
    },
    # ── Neighbors ─────────────────────────────────────────────────────────────
    "KNeighborsClassifier": {
        "task": "classification",
        "module": "neighbors",
        "desc": "K-Nearest Neighbors classifier.",
        "params": {
            "n_neighbors":    {"type": "int",   "default": 5,      "min": 1,     "desc": "Number of neighbors."},
            "weights":        {"type": "select","default": "uniform","options": ["uniform","distance"], "desc": "Weight function."},
            "algorithm":      {"type": "select","default": "auto", "options": ["auto","ball_tree","kd_tree","brute","ivfflat","ivfpq","ivfsq","approx","rbc"], "desc": "Algorithm for nearest neighbors search."},
            "leaf_size":      {"type": "int",   "default": 30,     "min": 1,     "desc": "Leaf size for ball_tree/kd_tree."},
            "metric":         {"type": "select","default": "minkowski","options": ["euclidean","minkowski","chebyshev","cosine","l2","l1"], "desc": "Distance metric."},
            "p":              {"type": "int",   "default": 2,      "min": 1,     "desc": "Minkowski p-parameter."},
            "n_jobs":         {"type": "int",   "default": 1,      "min": -1,    "desc": "Parallel jobs (-1 = all cores)."},
        },
    },
    "KNeighborsRegressor": {
        "task": "regression",
        "module": "neighbors",
        "desc": "K-Nearest Neighbors regressor.",
        "params": {
            "n_neighbors":    {"type": "int",   "default": 5,      "min": 1,     "desc": "Number of neighbors."},
            "weights":        {"type": "select","default": "uniform","options": ["uniform","distance"], "desc": "Weight function."},
            "algorithm":      {"type": "select","default": "auto", "options": ["auto","ball_tree","kd_tree","brute","ivfflat","ivfpq","ivfsq","approx","rbc"], "desc": "Algorithm for nearest neighbors search."},
            "leaf_size":      {"type": "int",   "default": 30,     "min": 1,     "desc": "Leaf size."},
            "metric":         {"type": "select","default": "minkowski","options": ["euclidean","minkowski","chebyshev","cosine","l2","l1"], "desc": "Distance metric."},
            "p":              {"type": "int",   "default": 2,      "min": 1,     "desc": "Minkowski p-parameter."},
        },
    },
    "NearestNeighbors": {
        "task": "unsupervised",
        "module": "neighbors",
        "desc": "Unsupervised nearest neighbors finder.",
        "params": {
            "n_neighbors":    {"type": "int",   "default": 5,      "min": 1,     "desc": "Number of neighbors."},
            "algorithm":      {"type": "select","default": "auto", "options": ["auto","ball_tree","kd_tree","brute","ivfflat","ivfpq","ivfsq","approx","rbc"], "desc": "Algorithm."},
            "leaf_size":      {"type": "int",   "default": 30,     "min": 1,     "desc": "Leaf size."},
            "metric":         {"type": "select","default": "minkowski","options": ["euclidean","minkowski","chebyshev","cosine","l2","l1"], "desc": "Distance metric."},
            "p":              {"type": "int",   "default": 2,      "min": 1,     "desc": "Minkowski p-parameter."},
            "n_jobs":         {"type": "int",   "default": 1,      "min": -1,    "desc": "Parallel jobs."},
        },
    },
    # ── Ensemble ──────────────────────────────────────────────────────────────
    "RandomForestClassifier": {
        "task": "classification",
        "module": "ensemble",
        "desc": "Random Forest classifier (GPU-accelerated).",
        "params": {
            "n_estimators":   {"type": "int",   "default": 100,    "min": 1, "max": 10000, "desc": "Number of trees."},
            "max_depth":      {"type": "int",   "default": 16,     "min": 1, "max": 128,   "desc": "Max depth of each tree."},
            "min_samples_split": {"type": "int","default": 2,      "min": 2,               "desc": "Min samples to split a node."},
            "min_samples_leaf":  {"type": "int","default": 1,      "min": 1,               "desc": "Min samples in a leaf."},
            "max_features":   {"type": "select","default": "sqrt", "options": ["sqrt","log2","auto"], "desc": "Features to consider for best split."},
            "n_bins":         {"type": "int",   "default": 128,    "min": 2, "max": 256,   "desc": "Number of bins for histograms (cuML specific)."},
            "random_state":   {"type": "int",   "default": 42,     "desc": "Random seed."},
            "n_streams":      {"type": "int",   "default": 8,      "min": 1,               "desc": "Number of GPU streams."},
            "split_criterion":{"type": "select","default": "gini", "options": ["gini","entropy"], "desc": "Split criterion."},
        },
    },
    "RandomForestRegressor": {
        "task": "regression",
        "module": "ensemble",
        "desc": "Random Forest regressor (GPU-accelerated).",
        "params": {
            "n_estimators":   {"type": "int",   "default": 100,    "min": 1, "max": 10000, "desc": "Number of trees."},
            "max_depth":      {"type": "int",   "default": 16,     "min": 1, "max": 128,   "desc": "Max depth of each tree."},
            "min_samples_split": {"type": "int","default": 2,      "min": 2,               "desc": "Min samples to split a node."},
            "min_samples_leaf":  {"type": "int","default": 1,      "min": 1,               "desc": "Min samples in a leaf."},
            "max_features":   {"type": "select","default": "auto", "options": ["sqrt","log2","auto"], "desc": "Features to consider for best split."},
            "n_bins":         {"type": "int",   "default": 128,    "min": 2, "max": 256,   "desc": "Number of bins for histograms (cuML specific)."},
            "random_state":   {"type": "int",   "default": 42,     "desc": "Random seed."},
            "n_streams":      {"type": "int",   "default": 8,      "min": 1,               "desc": "Number of GPU streams."},
            "split_criterion":{"type": "select","default": "mse",  "options": ["mse","mae"], "desc": "Split criterion."},
        },
    },
    # ── Preprocessing (can be used as transformer in pipeline) ────────────────
    "StandardScaler": {
        "task": "preprocessing",
        "module": "preprocessing",
        "desc": "Standardize features (zero mean, unit variance).",
        "params": {
            "copy":           {"type": "bool",  "default": True,   "desc": "Copy input data."},
            "with_mean":      {"type": "bool",  "default": True,   "desc": "Subtract mean."},
            "with_std":       {"type": "bool",  "default": True,   "desc": "Scale to unit variance."},
        },
    },
    "MinMaxScaler": {
        "task": "preprocessing",
        "module": "preprocessing",
        "desc": "Scale features to a given range (default [0, 1]).",
        "params": {
            "feature_range":  {"type": "tuple_float","default": [0.0, 1.0], "desc": "Desired output range."},
            "copy":           {"type": "bool",  "default": True,   "desc": "Copy input data."},
        },
    },
    # ── Manifold ──────────────────────────────────────────────────────────────
    "TSNE": {
        "task": "decomposition",
        "module": "manifold",
        "desc": "t-SNE dimensionality reduction for visualization.",
        "params": {
            "n_components":   {"type": "int",   "default": 2,      "min": 1, "max": 3, "desc": "Target dimensions."},
            "perplexity":     {"type": "float", "default": 30.0,   "min": 5.0, "max": 50.0, "desc": "Perplexity parameter."},
            "learning_rate":  {"type": "float", "default": 200.0,  "min": 10.0, "desc": "Learning rate."},
            "n_iter":         {"type": "int",   "default": 1000,   "min": 250,   "desc": "Number of iterations."},
            "random_state":   {"type": "int",   "default": 42,     "desc": "Random seed."},
            "method":         {"type": "select","default": "fft",  "options": ["fft","barnes_hut","exact"], "desc": "Gradient calculation method."},
            "angle":          {"type": "float", "default": 0.5,    "min": 0.0, "max": 1.0, "desc": "Barnes-Hut angle (tradeoff speed vs accuracy)."},
        },
    },
    "UMAP": {
        "task": "decomposition",
        "module": "manifold",
        "desc": "Uniform Manifold Approximation and Projection.",
        "params": {
            "n_components":   {"type": "int",   "default": 2,      "min": 1,     "desc": "Target dimensions."},
            "n_neighbors":    {"type": "int",   "default": 15,     "min": 2,     "desc": "Number of neighbors for local manifold approximation."},
            "min_dist":       {"type": "float", "default": 0.1,    "min": 0.0,   "desc": "Min distance between embedded points."},
            "spread":         {"type": "float", "default": 1.0,    "min": 0.0,   "desc": "Effective scale of embedded points."},
            "learning_rate":  {"type": "float", "default": 1.0,    "min": 0.0,   "desc": "Initial learning rate."},
            "n_epochs":       {"type": "int",   "default": None,   "min": 1, "nullable": True, "desc": "Number of training epochs (None = auto)."},
            "random_state":   {"type": "int",   "default": 42,     "desc": "Random seed."},
            "metric":         {"type": "select","default": "euclidean","options": ["euclidean","l2","l1","manhattan","cosine","correlation"], "desc": "Distance metric."},
            "init":           {"type": "select","default": "spectral","options": ["spectral","random"], "desc": "Embedding initialization."},
        },
    },
}

# ── Metric definitions per task type ──────────────────────────────────────────
TASK_METRICS = {
    "classification": ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc"],
    "regression":     ["r2", "neg_mean_squared_error", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    "clustering":     ["silhouette", "calinski_harabasz", "davies_bouldin"],
    "decomposition":  ["explained_variance_ratio"],
    "preprocessing":  [],
    "unsupervised":   [],
}

# Higher-is-better flag per metric
METRIC_HIGHER_BETTER = {
    "accuracy": True,
    "f1_weighted": True,
    "precision_weighted": True,
    "recall_weighted": True,
    "roc_auc": True,
    "r2": True,
    "neg_mean_squared_error": False,
    "neg_mean_absolute_error": False,
    "neg_root_mean_squared_error": False,
    "silhouette": True,
    "calinski_harabasz": True,
    "davies_bouldin": False,
    "explained_variance_ratio": True,
}
