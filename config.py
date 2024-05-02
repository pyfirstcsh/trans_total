"""Switch Transformers model configuration"""

from transformers import T5Config


class SwitchConfig(T5Config):
    def __init__(
        self,
        expert_capacity=64,
        num_experts=4,
        router_bias=False,
        router_dtype="float32",
        router_jitter_noise=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_bias = router_bias
        self.router_dtype = router_dtype
        self.router_jitter_noise = router_jitter_noise

    def to_dict(self):
        output = super().to_dict()
        output["num_experts"] = self.num_experts
        output["expert_capacity"] = self.expert_capacity
        output["router_bias"] = self.router_bias
        output["router_dtype"] = self.router_dtype
        output["router_jitter_noise"] = self.router_jitter_noise
        return output

    @classmethod
    def from_dict(cls, config_dict):
        # 确保新参数有默认值
        config_dict.setdefault("num_experts", 4)
        config_dict.setdefault("expert_capacity", 64)
        config_dict.setdefault("router_bias", False)
        config_dict.setdefault("router_dtype", "float32")
        config_dict.setdefault("router_jitter_noise", 0.01)

        return cls(**config_dict)


class SparseConfig(T5Config):
    def __init__(
        self,
        d_lowrank=32,
        N=64,
        c_temperature=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_lowrank = d_lowrank
        self.N = N
        self.c_temperature = c_temperature

    def to_dict(self):
        output = super().to_dict()
        output["d_lowrank"] = self.d_lowrank
        output["N"] = self.N
        output["c_temperature"] = self.c_temperature
        return output

    @classmethod
    def from_dict(cls, config_dict):
        # 确保新参数有默认值
        config_dict.setdefault("d_lowrank", 32)
        config_dict.setdefault("N", 64)
        config_dict.setdefault("c_temperature", 1.0)
        return cls(**config_dict)


class MultConvAttnConfig(T5Config):
    def __init__(
        self,
        S=8,
        M=64,
        F=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.S = S
        self.M = M
        self.F = F

    def to_dict(self):
        output = super().to_dict()
        output["S"] = self.S
        output["M"] = self.M
        output["F"] = self.F
        return output

    @classmethod
    def from_dict(cls, config_dict):
        # 确保新参数有默认值
        config_dict.setdefault("S", 8)
        config_dict.setdefault("M", 64)
        config_dict.setdefault("F", 3)
        return cls(**config_dict)
