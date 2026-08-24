from abc import ABC, abstractmethod


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states,
        w1,
        w2,
        gating_output,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ): ...
