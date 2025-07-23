from safetensors import safe_open
from torch.nn import Linear
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import HQQLinear
from hqq.core.quantize import BaseQuantizeConfig
import resource
from transformers import AutoConfig

quant_config = BaseQuantizeConfig(nbits=4, group_size=64)

model_config = AutoConfig.from_pretrained("./models/Qwen2-0.5B/")
file_path = "./models/Qwen2-0.5B/model.safetensors"
save_dir = "./models/Qwen2-0.5B-layers"
weights = {}
module_name = "model.layers.0.mlp.gate_proj"
num_layers = model_config.num_hidden_layers

layers_linear_dict = [
    ("self_attn.q_proj", True),
    ("self_attn.k_proj", True),
    ("self_attn.v_proj", True),
    ("self_attn.o_proj", False),
    ("mlp.gate_proj", False),
    ("mlp.up_proj", False),
    ("mlp.down_proj", False),
]

layers_other = ["input_layernorm", "post_attention_layernorm"]

other_dict = ["model.embed_tokens", "model.norm"]

with safe_open(file_path, framework="pt") as f:
    for i in range(num_layers):
        for name, has_bias in layers_linear_dict:
            module_name = f"model.layers.{i}.{name}"
            weight = f.get_tensor(module_name + ".weight")
            out_feat, in_feat = weight.shape
            linear_layer = Linear(in_feat, out_feat, bias=has_bias)
            if has_bias:
                bias = f.get_tensor(module_name + ".bias")
                Linear.load_state_dict(linear_layer, {"weight": weight, "bias": bias})
            else:
                Linear.load_state_dict(linear_layer, {"weight": weight})
            qmodule = HQQLinear(linear_layer, quant_config, device="cpu")
            qmodule.encoded_state_dict = False
            weights[module_name] = dict(qmodule.state_dict())

        for name in layers_other:
            module_name = f"model.layers.{i}.{name}"
            weight = f.get_tensor(module_name + ".weight")
            weights[module_name] = {"weight": weight}

    for name in other_dict:
        weight = f.get_tensor(name + ".weight")
        weights[name] = {"weight": weight}

    weights["lm_head"] = weights["model.embed_tokens"]

max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Max memory used: {max_mem_used} KB")
AutoHQQHFModel.save_weights(weights, save_dir)
