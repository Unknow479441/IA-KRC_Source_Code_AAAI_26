import numpy as np
import torch as th

class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError

class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):

        if tensor.min() < 0 or tensor.max() >= self.out_dim:
            import warnings
            warnings.warn(
                f"[OneHot] Received indices outside the valid range [0, {self.out_dim - 1}]. "
                f"Clamping them to the nearest valid value.")
            tensor = tensor.clamp(0, self.out_dim - 1)

        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32