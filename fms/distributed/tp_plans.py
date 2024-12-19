from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
   parallelize_module,
   ColwiseParallel,
   RowwiseParallel,
   SequenceParallel,
   PrepareModuleInput,
   PrepareModuleOutput
)

tp_plans_module = {
    'granite': {
                "head": ColwiseParallel(output_layouts=Replicate(),),
                "base_model.embedding": RowwiseParallel(input_layouts=Replicate()),
            },
    'gpt_bigcode': {
                "head": ColwiseParallel(output_layouts=Replicate(),),
                "base_model.embedding": RowwiseParallel(input_layouts=Replicate()),
            },
}

tp_plans_layer = {
    'granite': {
                "attn.in_proj.qkv_fused": ColwiseParallel(),
                "attn.in_proj.query": ColwiseParallel(),
                "attn.in_proj.key": ColwiseParallel(),
                "attn.in_proj.value": ColwiseParallel(),
                "attn.dense": RowwiseParallel(),
                "ff_sub_layer.wg": ColwiseParallel(),
                "ff_sub_layer.wg1_fused": ColwiseParallel(),
                "ff_sub_layer.w2": RowwiseParallel(),
                "ff_sub_layer.w1": ColwiseParallel(),
                },
    'gpt_bigcode': {
                "attn.in_proj.qkv_fused": ColwiseParallel(),
                "attn.in_proj.query": ColwiseParallel(),
                "attn.in_proj.key": ColwiseParallel(),
                "attn.in_proj.value": ColwiseParallel(),
                "attn.dense": RowwiseParallel(),
                "ff_sub_layer.w2": RowwiseParallel(),
                "ff_sub_layer.w1": ColwiseParallel(),
                },
}

def get_tp_plan_module(specified_model):
    if specified_model not in tp_plans_module:
        raise ValueError(f"Specified model: {specified_model} does not exist in tp_plans_module")
    return tp_plans_module[specified_model]

def get_tp_plan_layer(specified_model):
    if specified_model not in tp_plans_layer:
        raise ValueError(f"Specified model: {specified_model} does not exist in tp_plans_layer")
    return tp_plans_layer[specified_model]
