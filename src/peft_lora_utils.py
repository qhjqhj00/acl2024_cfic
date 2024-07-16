# coding=utf-8
import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
import torch
from peft import set_peft_model_state_dict, PeftModel
import deepspeed

PEFT_ADAPTER_WEIGHTS_NAME = "adapter_model.bin"


def _save_lora_adapter_model(model: PeftModel,
                             save_dir: str):
    gather_v = [v for k, v in model.named_parameters() if "lora" in k]
    with deepspeed.zero.GatheredParameters(gather_v):
        model.save_pretrained(save_dir)


class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        _save_lora_adapter_model(model=kwargs["model"],
                                 save_dir=checkpoint_folder)
        return control


class SavePeftModelAtEndCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        _save_lora_adapter_model(model=kwargs["model"],
                                 save_dir=args.output_dir)
        return control


def resume_lora_model_from_checkpoint(model,
                                      checkpoint: str):
    if not isinstance(model, PeftModel):
        raise ValueError(f"Input model is not a PeftModel.")

    checkpoint_name = os.path.join(checkpoint, PEFT_ADAPTER_WEIGHTS_NAME)
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(checkpoint, "pytorch_model.bin")

    if os.path.exists(checkpoint_name):
        print(f"Resume lora model from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        adapters_weights = {k: v for k, v in adapters_weights.items() if 'lora' in k}
        print(f"Resume {len(adapters_weights)} lora layers")
        for k, v in adapters_weights.items():
            with deepspeed.zero.GatheredParameters(model.get_parameter(k), modifier_rank=0):
                model.get_parameter(k).data = v
    else:
        raise ValueError(f"Checkpoint {checkpoint_name} not found")


def merge_lora_checkpoint_weights(checkpoint_dir: str,
                                  save_dir: str,
                                  target_modules: [str],
                                  lora_dim: int,
                                  lora_alpha: float,
                                  adapter_name: str = 'default'
                                  ):
    checkpoint_name = os.path.join(checkpoint_dir, "pytorch_model.bin")
    save_name = os.path.join(save_dir, "pytorch_model.bin")
    scaling = lora_alpha / lora_dim
    PEFT_PREFIX = 'base_model.model.'
    WEIGHT_SUFFIX = '.weight'

    new_state_dict = {}
    if os.path.exists(checkpoint_name):
        state_dict = torch.load(checkpoint_name)
        for key in state_dict.keys():
            if any(key.endswith(target_key + ".weight") for target_key in target_modules):
                lora_a_key = key[:-len(WEIGHT_SUFFIX)] + ".lora_A." + adapter_name + WEIGHT_SUFFIX
                lora_b_key = key[:-len(WEIGHT_SUFFIX)] + ".lora_B." + adapter_name + WEIGHT_SUFFIX
                state_dict[key].data += (state_dict[lora_b_key] @ state_dict[lora_a_key]) * scaling
                new_state_dict[key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key] = state_dict[key]
            elif "lora" not in key:
                new_state_dict[key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key] = state_dict[key]
        torch.save(new_state_dict, save_name)
    else:
        raise ValueError(f"Checkpoint {checkpoint_name} not found")
