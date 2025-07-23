from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *
import torch

model_path = "allenai/OLMoE-1B-7B-0924"
save_dir = "./models/OLMoE-1B-7B-0924-HQQ-4bit-flash"

num_experts = 64
num_layers = 16

q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
q8_config = BaseQuantizeConfig(nbits=8, group_size=64)

experts_quant_config = {}
for i in range(num_layers):
    for j in range(num_experts):
        config = q4_config if j < num_experts // 2 else q8_config
        experts_quant_config[f"model.layers.{i}.mlp.experts.{j}.gate_proj"] = config
        experts_quant_config[f"model.layers.{i}.mlp.experts.{j}.up_proj"] = config
        experts_quant_config[f"model.layers.{i}.mlp.experts.{j}.down_proj"] = config

quant_config = {
    "self_attn.q_proj": q4_config,
    "self_attn.k_proj": q4_config,
    "self_attn.v_proj": q4_config,
    "self_attn.o_proj": q4_config,
    "mlp.gate": q8_config,
    **experts_quant_config,
}

quant_config = q4_config

# Load the model on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
AutoHQQHFModel.quantize_model(model, quant_config=quant_config)

AutoHQQHFModel.save_quantized(model, save_dir)
tokenizer.save_pretrained(save_dir)
