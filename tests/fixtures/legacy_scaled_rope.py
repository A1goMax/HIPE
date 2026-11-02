"""Reference from PE 060b6a6, retained only for numerical equivalence tests."""
import torch
from typing import Optional
from hipe.olmo.config import ModelConfig
from hipe.olmo.model import RotaryEmbedding, BufferCache, _non_meta_init_device

class LegacyScaledRotaryEmbedding(RotaryEmbedding):
    """
    Scaled RoPE with Optional Layer-wise Control.
    Modes:
    1. Uniform Scaling: All layers use the same sigma (if threshold < 0).
    2. Bio-Gradient: Shallow layers use Standard RoPE, Deep layers use Scaled RoPE (if threshold >= 0).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache, sigma: float = 1.0, layer_index: Optional[int] = None):
        """
        Args:
            sigma (float): The scaling factor.
            layer_index (int, optional): The current layer index.
        """
        super().__init__(config, cache)

        # --- 1. 更加健壮的层级控制逻辑 ---

        # 默认行为：启用 Scaling (即假设我们想要全层应用 Sigma)
        use_scaling = True

        # 获取阈值。关键修改：默认值设为 -1 (代表禁用层级策略，保持 Uniform)
        # 只有当你在 config 中显式设置了 rope_scaling_threshold >= 0 时，才会激活HIPE
        scaling_threshold = getattr(config, "rope_scaling_threshold", -1)

        # 仅当阈值有效(>=0) 且 当前层号已知 且 当前层 <= 阈值 时，才强制关闭 Scaling
        if scaling_threshold >= 0 and layer_index is not None:
            if layer_index <= scaling_threshold:
                use_scaling = False
                print(f"Layer {layer_index}: Scaling DISABLED (Gradient Mode)")

        # 否则，use_scaling 保持为 True，即执行原来的逻辑（全层使用传入的 sigma）

        # --------------------------------

        # 获取维度信息
        dim = self.config.d_model // self.config.n_heads

        # --- 2. 如果不启用 Scaling (即退化为 Standard RoPE) ---
        # 这是针对需要layer分布的前几层，前几层需要使用原始的RoPE
        if not use_scaling:
            scale_full = torch.ones(self.config.n_heads, dim)
            self.register_buffer('scale_factor', scale_full)
            return

        # --- 3. 如果启用 Scaling (应用 Sigma) ---

        if isinstance(sigma, float):
            sigmas = [sigma] * self.config.n_heads
        else:
            sigmas = sigma

        if len(sigmas) != self.config.n_heads:
             raise ValueError(f"Sigma count ({len(sigmas)}) must match head count ({self.config.n_heads})")

        with torch.no_grad():
            device = _non_meta_init_device(config)
            inv_freq = self.get_inv_freq(device)

            sigmas_tensor = torch.tensor(sigmas, device=device, dtype=torch.float).view(self.config.n_heads, 1)
            freqs = inv_freq.view(1, -1)

            decay_func = getattr(self.config, 'decay_func', 'gaussian')

            if decay_func == 'gaussian':
                scale = torch.exp(-sigmas_tensor**2 * freqs**2/2) * freqs
            elif decay_func == 'exp':
                scale = (1/sigmas_tensor)**2 / ((1/sigmas_tensor)**2 + freqs**2) * freqs
            elif decay_func == 'power':
                scale = torch.exp(-sigmas_tensor * freqs) * freqs
            elif decay_func == 'segmented':
                order = getattr(self.config, 'decay_order', 8)
                scale = (1.0 / (1.0 + (sigmas_tensor * freqs) ** order)) * freqs
            else:
                scale = torch.exp(-sigmas_tensor**2 * freqs**2/2) * freqs

            scale = torch.sqrt(scale)
            scale_full = torch.cat((scale, scale), dim=-1)

            correction_factor = torch.rsqrt(torch.mean(scale_full**2))
            scale_full = scale_full * correction_factor

            self.register_buffer('scale_factor', scale_full)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * self.scale_factor.view(1, self.config.n_heads, 1, -1)
        return super().apply_rotary_pos_emb(pos_sin, pos_cos, t_scaled)
