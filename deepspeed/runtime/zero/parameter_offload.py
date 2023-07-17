# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import OrderedDict, abc
import sys
from typing import Mapping, Optional, Sequence, Union
import torch
from collections import OrderedDict
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import _init_external_params
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_coordinator import (
    PartitionedParameterCoordinator,
    InflightParamRegistry,
    iter_params,
)
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

FWD_MODULE_STACK = list()


def is_builtin_type(obj):
    # https://stackoverflow.com/a/17795199
    return (
        obj.__class__.__module__ == "__builtin__"
        or obj.__class__.__module__ == "builtins"
    )


def isinstance_namedtuple(obj: object) -> bool:
    """
    Is this an instance of namedtuple/NamedTuple?
    From: https://stackoverflow.com/a/62692640

    Args:
        obj (object): An object.

    Returns:
        bool: True if namedtuple/NamedTuple else False.
    """
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


# ensure we only warn once, otherwise every iteration will trigger a warning
warned = False


def _apply_to_tensors_only(module, functional, backward_function, outputs):
    """
    Apply a torch.autograd.Function that calls a `backward_function` to every Tensor in `outputs`.

    Args:
        module (torch.nn.Module):  A torch module
        functional (Type[torch.autograd.Function]): The function class to apply.
        backward_function (Callable[[torch.nn.Module], None]): A backward_function to pass to
            `functional.apply`.
        outputs (Any): The output of `module`.

    Returns:
        Any: The output of `module`.
    """
    if isinstance(outputs, (tuple, list)):
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(
                module, functional, backward_function, output
            )
            touched_outputs.append(touched_output)

        if isinstance_namedtuple(outputs):
            # namedtuples require a slightly different syntax.
            return outputs.__class__(*touched_outputs)

        return outputs.__class__(touched_outputs)
    elif isinstance(outputs, dict):
        # apply inplace to avoid recreating dict inherited objects
        for key in outputs.keys():
            outputs[key] = _apply_to_tensors_only(
                module, functional, backward_function, outputs[key]
            )
        return outputs

    elif isinstance(outputs, torch.Tensor):
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_outputs = functional.apply(module, backward_function, outputs)

        # restore zero param attributes if those get stripped by `backward_function`
        if not is_zero_param(touched_outputs) and is_zero_param(outputs):
            touched_outputs.ds_param_alias = outputs
        return touched_outputs
    else:
        if not is_builtin_type(outputs):
            global warned
            if not warned and dist.get_rank() == 0:
                logger.warning(
                    f"A module has unknown inputs or outputs type ({type(outputs)}) and the tensors embedded in it cannot be detected. "
                    "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and "
                    "output tensors and therefore may not get triggered properly."
                )
                warned = True
        return outputs


# for each tensor in outputs run the forward_function and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(
    module, forward_function, backward_function, outputs
):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(
                module, forward_function, backward_function, output
            )
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            outputs.register_hook(backward_function)
        return outputs
    else:
        return outputs


class ZeROOrderedDict(OrderedDict):
    def __init__(self, parent_module, *args, **kwargs):
        """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

        Args:
            parent_module (``collections.OrderedDict``): the collection to replace
        """

        super().__init__(*args, **kwargs)
        self._parent_module = parent_module
        self._in_forward = False

    def __getitem__(self, key):
        param = super().__getitem__(key)

        # Params can be registered as None (e.g., bias)
        if param is None:
            return param

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if self._parent_module._parameters._in_forward:
                register_external_parameter(FWD_MODULE_STACK[-1], param)
                param.all_gather()
                print_rank_0(
                    f"Registering external parameter from getter {key} ds_id = {param.ds_id}",
                    force=False,
                )

        return param


def _inject_parameters(module, cls):
    for module in module.modules():
        if cls == ZeROOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()

        for key, param in module._parameters.items():
            new_param[key] = param
        module._parameters = new_param


class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, "applied_pre_backward_ref_cnt"):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        # print(f"After Forward: {ctx.module.__class__.__name__}")
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # print(f"Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        if output.requires_grad:
            # TODO SOME TIMES post backward does not seem to be triggered debug in detail
            # Should only cause increase in memory not correctness issue
            # if output.grad_fn.__class__.__name__ == 'ViewBackward':
            #    ctx.view=True
            #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
            # assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
            # if module.ds_grads_remaining == 0:
            #    print(f"Before Forward: {ctx.module.__class__.__name__}")
            module.ds_grads_remaining += 1
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output

    @staticmethod
    def backward(ctx, *args):
        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        if ctx.module.ds_grads_remaining == 0:
            ctx.pre_backward_function(ctx.module)
            # print(f"After Backward: {ctx.module.__class__.__name__}")
        return (None, None) + args


def _recursively_apply_to_tensors_only(module, function, outputs):
    if isinstance(outputs, (tuple, list)):
        touched_outputs = []
        for output in outputs:
            touched_output = _recursively_apply_to_tensors_only(
                module, function, output
            )
            touched_outputs.append(touched_output)
        return outputs.__class__(touched_outputs)
    elif isinstance(outputs, dict):
        # apply inplace to avoid recreating dict inherited objects
        for key in outputs.keys():
            outputs[key] = _recursively_apply_to_tensors_only(
                module, function, outputs[key]
            )
        return outputs

    elif type(outputs) is torch.Tensor:
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_outputs = function(outputs)

        # restore zero param attributes if those get stripped by `backward_function`
        if not is_zero_param(touched_outputs) and is_zero_param(outputs):
            touched_outputs.ds_param_alias = outputs
        return touched_outputs
    else:
        if not is_builtin_type(outputs):
            global warned
            if not warned and dist.get_rank() == 0:
                logger.warning(
                    f"A module has unknown inputs or outputs type ({type(outputs)}) and the tensors embedded in it cannot be detected. "
                    "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and "
                    "output tensors and therefore may not get triggered properly."
                )
                warned = True
        return outputs


module_to_param_dict = {}


class ORTPreForwardwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        offload,
        module,
        pre_forward_function,
        post_backward_function,
        input_count,
        partition_param_count,
        kwarg_tensor_count,
        *inputs_and_partitioned_parms,
    ):
        offload.pre_sub_module_forward_function(module)

        inputs = inputs_and_partitioned_parms[:input_count]

        # the first time MUST be run with PyTorch (export to onnx)
        global module_to_param_dict
        if module not in module_to_param_dict:
            module_to_param_dict[module] = inputs_and_partitioned_parms[
                input_count : partition_param_count + input_count
            ]

        partitioned_params = module_to_param_dict[module]

        kwarg_tensors = inputs_and_partitioned_parms[
            partition_param_count + input_count :
        ]
        assert len(kwarg_tensors) == kwarg_tensor_count

        ctx.module = module
        ctx.offload = offload
        ctx.post_backward_function = post_backward_function
        ctx.partitioned_params = partitioned_params
        ctx.kwarg_tensor_count = kwarg_tensor_count

        module.ds_grads_remaining = 0

        # if input_count == 0:
        #     ctx.num_of_input = 0
        #     return None

        # def func(input_):
        #     if input_.requires_grad:
        #         # TODO SOME TIMES post backward does not seem to be triggered debug in detail
        #         # Should only cause increase in memory not correctness issue
        #         # if output.grad_fn.__class__.__name__ == 'ViewBackward':
        #         #    ctx.view=True
        #         #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
        #         # assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
        #         # if module.ds_grads_remaining == 0:
        #         #    print(f"Before Forward: {ctx.module.__class__.__name__}")
        #         module.ds_grads_remaining += 1
        #     return input_.detach().requires_grad_(input_.requires_grad)

        # rets = _recursively_apply_to_tensors_only(module, func, inputs)
        # if not isinstance(rets, (tuple, list, dict, torch.Tensor)):
        #     input_count = 0
        #     rets = ()
        # elif isinstance(rets, torch.Tensor):
        #     input_count = 1
        #     rets = (rets,)
        # else:
        #     input_count = len(rets)

        rets = ()
        rets += tuple([pre_forward_function(module, input_) for input_ in inputs])

        ctx.num_of_input = input_count
        # if input_count == 0:
        #     # pengwa: need return something instead of empty list, otherwise, torch export explains "Couldn't lower all tuples. prim::PythonOp"
        #     rets = partitioned_params
        #     for a in partitioned_params:
        #         print(">>a.size(): ", a.size())
        # else:
        #     rets += partitioned_params
        #     for a in partitioned_params:
        #         print("<<a.size(): ", a.size())
        rets += partitioned_params
        rets += kwarg_tensors
        assert len(rets) != 0
        # if len(rets) == 0:
        #     return None

        # if len(rets) == 1:
        #     return rets[0]
        return rets

    @staticmethod
    def backward(ctx, *args):
        # ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        # if ctx.module.ds_grads_remaining == 0:
        ctx.module.ds_grads_remaining = 0  # pengwa

        # for i, p in enumerate(ctx.partitioned_params):
        #     if p.grad is None:
        #         p.grad = args[ctx.num_of_input + i]
        #     else:
        #         p.grad += args[ctx.num_of_input + i]

        if ctx.num_of_input > 0:
            # print(f"Before Backward: {ctx.module.__class__.__name__}")
            ctx.post_backward_function(ctx.offload, ctx.module)

        return (
            (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            + args[: ctx.num_of_input]
            + (None,) * (len(ctx.partitioned_params) + ctx.kwarg_tensor_count)
        )


def _recursively_tensor_apply_func(function, outputs):
    if isinstance(outputs, (tuple, list)):
        touched_outputs = []
        for output in outputs:
            touched_output = _recursively_tensor_apply_func(function, output)
            touched_outputs.append(touched_output)
        return outputs.__class__(touched_outputs)
    elif isinstance(outputs, dict):
        # apply inplace to avoid recreating dict inherited objects
        for key in outputs.keys():
            outputs[key] = _recursively_tensor_apply_func(function, outputs[key])
        return outputs

    elif type(outputs) is torch.Tensor:
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_outputs = function(outputs)
        return touched_outputs
    else:
        if not is_builtin_type(outputs):
            global warned
            if not warned and dist.get_rank() == 0:
                logger.warning(
                    f"A module has unknown inputs or outputs type ({type(outputs)}) and the tensors embedded in it cannot be detected. "
                    "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and "
                    "output tensors and therefore may not get triggered properly."
                )
                warned = True
        return outputs


class ORTPostForwardwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        post_forward_function,
        pre_backward_function,
        input_count,
        output_count,
        *inputs_and_outputs,
    ):
        inputs = inputs_and_outputs[:input_count]
        outputs = inputs_and_outputs[input_count:]
        post_forward_function(module, inputs, outputs)

        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        ctx.input_count = input_count
        if not hasattr(module, "applied_pre_backward_ref_cnt"):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        # print(f"After Forward: {ctx.module.__class__.__name__}")

        rets = _recursively_tensor_apply_func(
            lambda x: x.detach().requires_grad_(x.requires_grad)
            if x is not None
            else None,
            outputs,
        )

        print(
            "output of module " + module.__class__.__name__ + ": ",
            [a.size() if isinstance(a, torch.Tensor) else a for a in rets],
        )
        return rets
        # if isinstance(outputs, torch.Tensor):
        #     return outputs.detach()
        # if isinstance(outputs, (tuple, list)):
        #     outputs = [output.detach().requires_grad_(output.requires_grad) if output is not None else None for output in outputs]
        #     if output_count == 1:
        #     #     print(a)
        #     #     assert len(a) == 1
        #         outputs = outputs[0]
        #     return outputs
        # else:
        #     raise RuntimeError("fail handling output of forward")

    @staticmethod
    def backward(ctx, *args):
        # print(f"Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None, None, None, None) + (None,) * ctx.input_count + args


class ORTFinalForwardCleanupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offload, is_grad_enabled):
        if not is_grad_enabled:
            offload.get_param_coordinator(training=False).reset_step()
        return None

    @staticmethod
    def backward(ctx, *args):
        return (None, None)


class DeepSpeedZeRoOffload(object):
    def __init__(
        self,
        module,
        timers,
        ds_config,
        overlap_comm=True,
        prefetch_bucket_size=50000000,
        max_reuse_distance=1000000000,
        max_live_parameters=1000000000,
        param_persistence_threshold=100000,
        model_persistence_threshold=sys.maxsize,
        offload_param_config=None,
        mpu=None,
        zero_param_parallel_group=None,
        zero_quantized_weights=False,
    ):
        see_memory_usage("DeepSpeedZeRoOffload initialize [begin]", force=True)

        print_rank_0(
            f"initialized {__class__.__name__} with args: {locals()}", force=False
        )

        self.module = module
        self.timers = timers
        self.dtype = list(module.parameters())[0].dtype
        self.offload_device = None
        self.offload_param_pin_memory = False
        self.zero_param_parallel_group = zero_param_parallel_group
        self.zero_quantized_weights = zero_quantized_weights

        if (
            offload_param_config is not None
            and offload_param_config.device != OffloadDeviceEnum.none
        ):
            self.offload_device = offload_param_config.device
            self.offload_param_pin_memory = offload_param_config.pin_memory

        self._convert_to_zero_parameters(ds_config, module, mpu)

        for m in module.modules():
            _init_external_params(m)

        _inject_parameters(module, ZeROOrderedDict)

        self.param_numel_persistence_threshold = int(param_persistence_threshold)
        self.model_persistence_threshold = int(model_persistence_threshold)
        self.persistent_parameters = self.mark_persistent_parameters(
            self.param_numel_persistence_threshold, self.model_persistence_threshold
        )

        self.param_coordinators = {}
        self._prefetch_bucket_sz = int(prefetch_bucket_size)
        self._max_reuse_distance_in_numel = int(max_reuse_distance)
        self._max_available_parameters_in_numel = int(max_live_parameters)
        self.__allgather_stream = (
            get_accelerator().Stream()
            if overlap_comm
            else get_accelerator().default_stream()
        )

        if not hasattr(module, "ds_inflight_param_registry"):
            module.ds_inflight_param_registry = dict()
            # we need two registries, one for training and one for eval. They will be used when creating PartitionedParameterCoordinator
            module.ds_inflight_param_registry[True] = InflightParamRegistry()
            module.ds_inflight_param_registry[False] = InflightParamRegistry()
        self.__inflight_param_registry = module.ds_inflight_param_registry

        self.forward_hooks = []
        self.backward_hooks = []
        self.setup_zero_stage3_hooks()
        print_rank_0(
            f"Created module hooks: forward = {len(self.forward_hooks)}, backward = {len(self.backward_hooks)}",
            force=False,
        )

        see_memory_usage("DeepSpeedZeRoOffload initialize [end]", force=True)

    @instrument_w_nvtx
    def partition_all_parameters(self):
        """Partitioning Parameters that were not partitioned usually if parameters
        of modules whose input parameters do not require grad computation do not
        trigger post call and will therefore will remain unpartitioned"""
        self.get_param_coordinator(training=self.module.training).release_and_reset_all(
            self.module
        )
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    def get_param_coordinator(self, training):
        if not training in self.param_coordinators:
            self.param_coordinators[training] = PartitionedParameterCoordinator(
                prefetch_bucket_sz=self._prefetch_bucket_sz,
                max_reuse_distance_in_numel=self._max_reuse_distance_in_numel,
                max_available_parameters_in_numel=self._max_available_parameters_in_numel,
                allgather_stream=self.__allgather_stream,
                inflight_param_registry=self.__inflight_param_registry[training],
                prefetch_nvme=self.offload_device == OffloadDeviceEnum.nvme,
                timers=self.timers,
            )

        return self.param_coordinators[training]

    def empty_partition_cache(self):
        self.partition_all_parameters()

    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                if mpu:
                    group = mpu.get_data_parallel_group()

                Init(
                    module=module,
                    data_parallel_group=group,
                    dtype=self.dtype,
                    config_dict_or_path=ds_config,
                    remote_device=self.offload_device,
                    pin_memory=self.offload_param_pin_memory,
                    mpu=mpu,
                    zero_param_parallel_group=self.zero_param_parallel_group,
                    zero_quantized_weights=self.zero_quantized_weights,
                )

    def destroy(self):
        self._remove_module_hooks()

    def _remove_module_hooks(self):
        num_forward_hooks = len(self.forward_hooks)
        num_backward_hooks = len(self.backward_hooks)

        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        print_rank_0(
            f"Deleted module hooks: forward = {num_forward_hooks}, backward = {num_backward_hooks}",
            force=False,
        )

    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0

        # reset step if in inference mode
        @instrument_w_nvtx
        def _end_of_forward_hook(module, *args):
            # if not torch._C.is_grad_enabled():
            #     self.get_param_coordinator(training=False).reset_step()

            ORTFinalForwardCleanupFunction.apply(self, torch._C.is_grad_enabled())

        # likely one of them should be enough but just to be safe
        self._register_hooks_recursively(self.module)
        self.module.register_forward_hook(_end_of_forward_hook)

        # Add top module to stack trace
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)
        print(f"Added {self.module.__class__} to FWD_MODULE_STACK")

    def mark_persistent_parameters(self, param_threshold, model_threshold):
        persistent_params = []
        total_persistent_parameters = 0
        params_count = 0
        for name, param in self.module.named_parameters(recurse=True):
            if param.ds_numel + total_persistent_parameters > model_threshold:
                continue

            if param.ds_numel <= param_threshold:
                params_count += 1
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.ds_numel

        print_rank_0(
            f"Parameter Offload: Total persistent parameters: {total_persistent_parameters} in {params_count} params",
            force=True,
        )

        return persistent_params

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        # print(f"{module.__class__} : {module.id}")

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        @instrument_w_nvtx
        def _pre_forward_module_hook(module, *args):
            self.pre_sub_module_forward_function(module)

        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):
            global FWD_MODULE_STACK
            print(
                f"pop module from FWD_MODULE_STACK: {module.__class__.__name__} : {module.id}"
            )
            FWD_MODULE_STACK.pop()
            if output is None:
                output = []
            elif not isinstance(output, (list, tuple)):
                if torch.is_tensor(output):
                    output = [output]
                else:
                    # print(f'got UNKNOWN type {type(output)}')
                    outputs = []
                    output = output if isinstance(output, dict) else vars(output)
                    for name, val in output.items():
                        if not name.startswith("__") and torch.is_tensor(val):
                            outputs.append(val)
                    output = outputs

            for item in filter(
                lambda item: is_zero_param(item) or hasattr(item, "ds_param_alias"),
                output,
            ):
                key = id(item) if hasattr(item, "ds_id") else id(item.ds_param_alias)
                actual_external_param = (
                    item if hasattr(item, "ds_id") else item.ds_param_alias
                )

                if not any(key in m._external_params for m in FWD_MODULE_STACK):
                    actual_external_param.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(
                        module_to_register, actual_external_param
                    )
                    print_rank_0(
                        f"Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {actual_external_param.ds_id}.",
                        force=False,
                    )

                    # It's possible that the parameter was already external to the completed module. If so, remove it the
                    # registration as it will be covered by the outer module instead.
                    if key in module._external_params:
                        print_rank_0(
                            f"  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {actual_external_param.ds_id}",
                            force=False,
                        )
                        unregister_external_parameter(module, actual_external_param)

                    actual_external_param.all_gather()

            self.post_sub_module_forward_function(module)

        def _pre_backward_module_hook(module, inputs, output):
            @instrument_w_nvtx
            def _run_before_backward_function(sub_module):
                # some models (e.g. Albert) may run multiple forwards on the same layer in a loop
                # before doing backwards, so each backward will need a pre-fetch - using reference
                # counting to support this scenario
                # print(f"COUNTER before: {sub_module.applied_pre_backward_ref_cnt}")
                if sub_module.applied_pre_backward_ref_cnt > 0:
                    self.pre_sub_module_backward_function(sub_module)
                    sub_module.applied_pre_backward_ref_cnt -= 1
                # print(f"COUNTER after: {sub_module.applied_pre_backward_ref_cnt}")

            return _apply_to_tensors_only(
                module, PreBackwardFunction, _run_before_backward_function, output
            )

        # This is an alternate to doing _post_backward_module_hook
        # it uses tensor.register_hook instead of using torch.autograd.Function
        def _alternate_post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            # print(f"Before Forward {module.__class__.__name__}")

            def _run_after_backward_hook(*unused):
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    # print(f"After backward {module.__class__.__name__}")
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                if input.requires_grad:
                    module.ds_grads_remaining += 1

            return _apply_forward_and_backward_to_tensors_only(
                module, _run_before_forward_function, _run_after_backward_hook, inputs
            )

        def _post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            @instrument_w_nvtx
            def _run_after_backward_function(sub_module):
                if sub_module.ds_grads_remaining == 0:
                    self.post_sub_module_backward_function(sub_module)

            return _apply_to_tensors_only(
                module, PostBackwardFunction, _run_after_backward_function, inputs
            )

        # Pre forward hook
        # self.forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))

        def _ort_run_after_backward_function(offload, sub_module):
            if sub_module.ds_grads_remaining == 0:
                self.post_sub_module_backward_function(sub_module)

        def _ort_pre_forward_module_hook(module, inputs, kwargs):
            from onnxruntime.training.ortmodule import ORTModule

            if isinstance(module, ORTModule):
                return inputs, kwargs

            # print("enter _ort_pre_forward_module_hook", module, module.training)

            # flatten_input_list = []
            # if isinstance(inputs, torch.Tensor):
            #     flatten_input_list = [inputs]
            #     print(
            #         "88888888888888888_ort_pre_forward_module_hook inputs type",
            #         type(inputs),
            #         type(module),
            #         inputs.dtype,
            #         inputs.size(),
            #     )
            # elif isinstance(inputs, (tuple, list)):
            #     print(
            #         "88888888888888888_ort_pre_forward_module_hook inputs type",
            #         [
            #             f"{type(ret)}-{str(ret.size()) + '-' + str(ret.dtype) + '-' + str(ret.requires_grad) if isinstance(ret, torch.Tensor) else ''}"
            #             for ret in inputs
            #         ],
            #         type(module),
            #     )
            #     if any(
            #         [not isinstance(ret, (torch.Tensor, type(None))) for ret in inputs]
            #     ):
            #         raise RuntimeError(
            #             "Found non-tensor or non-None input in the input list."
            #         )
            #     flatten_input_list = [ret for ret in inputs]
            # elif isinstance(inputs, dict):
            #     print(
            #         "88888888888888888_ort_pre_forward_module_hook inputs type",
            #         {key: type(ret) for key, ret in inputs.items()},
            #         type(module),
            #     )
            #     if any(
            #         [
            #             not isinstance(ret, (torch.Tensor, type(None)))
            #             for ret in inputs.values()
            #         ]
            #     ):
            #         raise RuntimeError(
            #             "Found non-tensor or non-None input in the input list."
            #         )

            #     flatten_input_list = [ret for ret in inputs.values()]
            # else:
            #     raise RuntimeError("Unsupported inputs type")

            params_to_fetch = frozenset(iter_params(module))
            partitioned_params = []
            for param in params_to_fetch:
                if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    partitioned_params.append(param)

            partitioned_param_count = len(partitioned_params)

            p_schema, p_flatten_tensors = _flatten_data_with_schema(inputs)
            p_count = len(p_flatten_tensors)

            k_schema, k_flatten_tensors = _flatten_data_with_schema(kwargs)
            k_count = len(k_flatten_tensors)

            # hook_input_count = len(flatten_input_list)

            print(
                "count of flatten input for ORTPreForwardwardFunction.apply: count of p_flatten_tensors:"
                + str(p_count)
                + ", count of partitioned params: "
                + str(partitioned_param_count)
                + ", count of kwargs: "
                + str(k_count)
            )

            def _ort_run_before_forward_function(m, input_):
                if input_.requires_grad:
                    # TODO SOME TIMES post backward does not seem to be triggered debug in detail
                    # Should only cause increase in memory not correctness issue
                    # if output.grad_fn.__class__.__name__ == 'ViewBackward':
                    #    ctx.view=True
                    #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
                    # assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
                    # if module.ds_grads_remaining == 0:
                    #    print(f"Before Forward: {ctx.module.__class__.__name__}")
                    m.ds_grads_remaining += 1
                return input_.detach().requires_grad_(input_.requires_grad)

            rets = ORTPreForwardwardFunction.apply(
                self,
                module,
                _ort_run_before_forward_function,
                _ort_run_after_backward_function,
                p_count,
                partitioned_param_count,
                k_count,
                *(p_flatten_tensors + partitioned_params + k_flatten_tensors),
            )

            p_rets = rets[:p_count]
            k_rets = rets[-k_count:]

            print(
                "count of flatten output from ORTPreForwardwardFunction.apply: "
                + str(len(rets))
            )

            assert rets is not None and len(rets) > 0

            # p_rets = _unflatten_data_from_schema(p_schema, p_rets)
            k_rets = _unflatten_data_from_schema(k_schema, k_rets)

            # print(inputs, kwargs)
            # print(rets[:hook_input_count], k_rets)

            # if rets is None:
            #     print(
            #         "_ort_pre_forward_module_hook complete for module "
            #         + module.__class__.__name__
            #         + " with no output"
            #     )

            #     return None, k_rets
            #     # return rets + flatten_input_list

            if len(p_rets) == 0:
                print(
                    "_ort_pre_forward_module_hook complete for module "
                    + module.__class__.__name__
                    + " with no output2"
                )
                if isinstance(p_rets, torch.Tensor):
                    p_rets = [p_rets]

                # if len(p_rets) != len(flatten_input_list):
                #     print(
                #         f"{len(p_rets)} != {len(flatten_input_list)}",
                #         [type(r) for r in flatten_input_list],
                #     )
                return p_rets, k_rets

            if isinstance(p_rets, torch.Tensor):
                print(
                    "88888888888888888_ort_pre_forward_module_hook output rets type",
                    type(p_rets),
                    type(module),
                )
            elif isinstance(p_rets, (tuple, list)):
                print(
                    "88888888888888888_ort_pre_forward_module_hook output rets type",
                    [
                        f"{type(ret)}-{str(ret.size()) + '-' + str(ret.dtype) + '-' + str(ret.requires_grad) if isinstance(ret, torch.Tensor) else ''}"
                        for ret in p_rets
                    ],
                    type(module),
                )
                if any(
                    [not isinstance(ret, (torch.Tensor, type(None))) for ret in p_rets]
                ):
                    print(
                        "888888888888888888888888#########################",
                        type(module),
                    )
            elif isinstance(p_rets, dict):
                print(
                    "88888888888888888_ort_pre_forward_module_hook output rets type",
                    {key: type(ret) for key, ret in p_rets.items()},
                    type(module),
                )
            if isinstance(p_rets, (tuple, list)) and len(p_rets) == 1:
                print(
                    "88888888888888888888 exist with single postional output ",
                    p_rets[0].size(),
                )
                print(k_rets)
                return p_rets, k_rets
            return p_rets, k_rets
            # # import traceback
            # # traceback.print_stack()
            # if input_count == 1:
            #     a = a[0]

            # # print("exit _ort_pre_forward_module_hook", module, module.training, inputs,  a)
            # return a

        self.forward_hooks.append(
            module.register_forward_pre_hook(
                _ort_pre_forward_module_hook, with_kwargs=True
            )
        )

        # Post forward hook
        # self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))

        def _ort_run_before_backward_function(sub_module):
            # print("enter _ort_run_before_backward_function", sub_module, sub_module.training)
            # some models (e.g. Albert) may run multiple forwards on the same layer in a loop
            # before doing backwards, so each backward will need a pre-fetch - using reference
            # counting to support this scenario
            # print(f"COUNTER before: {sub_module.applied_pre_backward_ref_cnt}")
            if sub_module.applied_pre_backward_ref_cnt > 0:
                self.pre_sub_module_backward_function(sub_module)
                sub_module.applied_pre_backward_ref_cnt -= 1
            # print(f"COUNTER after: {sub_module.applied_pre_backward_ref_cnt}")
            # print("exit _ort_run_before_backward_function ", sub_module.training)

        class _TensorStub:
            """Tensor stub class used to represent model's input or output"""

            def __init__(
                self,
                name: Optional[str] = None,
                dtype: Optional[str] = None,
                shape=None,
                shape_dims: Optional[int] = None,
                tensor_idx=None,
            ):
                self.name: Optional[str] = name
                self.dtype: Optional[str] = dtype
                self.shape = shape
                self.shape_dims: Optional[int] = shape_dims  # r.g. rank.
                self.tensor_idx = tensor_idx

        _ModelInputOutputSchemaType = Union[
            None,
            str,
            _TensorStub,
            Sequence["_ModelInputOutputSchemaType"],
            Mapping[str, "_ModelInputOutputSchemaType"],
        ]

        def _flatten_data_with_schema(data):
            flatten_tensor_data = []
            tensor_idx = [-1]

            def _flatten_from_data(data):
                if data is None:
                    return data
                elif isinstance(data, str):
                    return data
                elif isinstance(data, (int, bool, float)):
                    # tensor_idx[0] += 1
                    # flatten_tensor_data.append(data)
                    # return _TensorStub(
                    #     dtype=data, shape_dims=0, tensor_idx=tensor_idx[0]
                    # )
                    return data
                # Depth first traversal to iterate over the data to replace every tensor with a stub
                elif isinstance(data, torch.Tensor):
                    tensor_idx[0] += 1
                    flatten_tensor_data.append(data)
                    return _TensorStub(
                        dtype=str(data.dtype),
                        shape_dims=len(data.size()),
                        tensor_idx=tensor_idx[0],
                    )

                # Instead of replacing the tensor with a stub in the original user input, build the stubbed_schema
                # from scratch from the user input.
                stubbed_schema: Optional[_ModelInputOutputSchemaType] = None
                if isinstance(data, abc.Sequence):
                    sequence_type = type(data)
                    stubbed_schema = [_flatten_from_data(val) for val in data]
                    try:
                        # namedtuple can be created by passing the list sequence to method _make
                        stubbed_schema = sequence_type._make(stubbed_schema)
                    except AttributeError:
                        # If attribute error encountered, create the sequence directly
                        stubbed_schema = sequence_type(stubbed_schema)
                elif isinstance(data, abc.Mapping):
                    dict_type = type(data)
                    stubbed_schema = {
                        key: _flatten_from_data(data[key]) for key in data
                    }
                    stubbed_schema = dict_type(**stubbed_schema)
                else:
                    raise RuntimeError(f"Unsupported data type: {type(data)}")
                return stubbed_schema

            schemas = _flatten_from_data(data)
            return schemas, flatten_tensor_data

        def _unflatten_data_from_schema(
            schema: Optional[_ModelInputOutputSchemaType], outputs
        ):
            """Follows the schema to generate an output that is expected by the user"""
            import copy

            def _replace_stub_with_tensor_value(schema, outputs):
                # Recursively traverse across schema and replace all _TensorStub
                # with torch.Tensor values from outputs following output_idx

                if schema is None:
                    return None
                elif isinstance(schema, str):
                    return schema
                elif isinstance(schema, (int, bool, float)):
                    return schema
                elif isinstance(schema, _TensorStub):
                    out = outputs[schema.tensor_idx]
                    return out

                if isinstance(schema, abc.Sequence):
                    sequence_type = type(schema)
                    if hasattr(sequence_type, "_make"):  # namedtuple
                        sequence_type = type(schema)
                        schema = sequence_type._make(
                            _replace_stub_with_tensor_value(uo, outputs)
                            for uo in schema
                        )
                    else:
                        schema = sequence_type(
                            _replace_stub_with_tensor_value(uo, outputs)
                            for uo in schema
                        )
                elif isinstance(schema, abc.Mapping):
                    new_user_output = copy.copy(schema)
                    for key in sorted(schema):
                        new_user_output[key] = _replace_stub_with_tensor_value(
                            new_user_output[key], outputs
                        )
                    schema = new_user_output
                else:
                    raise RuntimeError(f"Unsupported data type: {type(schema)}")

                return schema

            user_output = _replace_stub_with_tensor_value(schema, outputs)
            return user_output

        def _ort_post_forward_module_hook(module, inputs, outputs):
            from onnxruntime.training.ortmodule import ORTModule

            if isinstance(module, ORTModule):
                return
            # print("enter _ort_post_forward_module_hook", module, module.training, outputs)
            input = inputs
            # output = outputs
            if isinstance(input, torch.Tensor):
                input = [input]

            assert isinstance(input, (tuple, list))
            # assert isinstance(output, (tuple, list))
            input_and_output = []
            for i in input:
                input_and_output.append(i)

            if not isinstance(outputs, (list, tuple, torch.Tensor)):
                print(
                    "99999999999999_ort_post_forward_module_hook output type is: ",
                    type(outputs),
                )

            schema, flatten_tensors = _flatten_data_with_schema(outputs)
            input_and_output.extend(flatten_tensors)

            # input_tensors, packed_non_tensors = split_non_tensors(input)
            rets = ORTPostForwardwardFunction.apply(
                module,
                _post_forward_module_hook,
                _ort_run_before_backward_function,
                len(input),
                len(flatten_tensors),
                *input_and_output,
            )
            # print("exit _ort_post_forward_module_hook", module, module.training)

            rets = _unflatten_data_from_schema(schema, rets)

            if isinstance(rets, torch.Tensor):
                print(
                    "99999999999999_ort_post_forward_module_hook output rets type",
                    type(rets),
                )
            elif isinstance(rets, (tuple, list)):
                print(
                    "99999999999999_ort_post_forward_module_hook output rets type",
                    [type(ret) for ret in rets],
                )
                if any(
                    [not isinstance(ret, (torch.Tensor, type(None))) for ret in rets]
                ):
                    print("9999999999999999999999#########################", module)
            elif isinstance(rets, dict):
                print(
                    "99999999999999_ort_post_forward_module_hook output rets type",
                    {key: type(ret) for key, ret in rets.items()},
                )

            # if isinstance(rets, (tuple, list)) and len(rets) == 1:
            #     return rets[0]
            return rets

        self.forward_hooks.append(
            module.register_forward_hook(_ort_post_forward_module_hook)
        )

        # # Pre backward hook
        # self.backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))

        # # post backward hook
        # self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))

    @torch.no_grad()
    def pre_sub_module_forward_function(self, sub_module):
        see_memory_usage(
            f"Before sub module function {sub_module.__class__.__name__}", force=False
        )

        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(sub_module)
        print(
            "append moduke to stack "
            + sub_module.__class__.__name__
            + " "
            + str(sub_module.id)
            + " "
            + str(len(FWD_MODULE_STACK))
        )
        # print("pre_sub_module_forward_function>> ", sub_module, len(FWD_MODULE_STACK))

        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module, forward=True)

        see_memory_usage(
            f"Before sub module function {sub_module.__class__.__name__} after fetch",
            force=False,
        )

    @torch.no_grad()
    def post_sub_module_forward_function(self, sub_module):
        see_memory_usage(
            f"After sub module function {sub_module.__class__.__name__} {sub_module.id} before release",
            force=False,
        )

        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.release_sub_module(sub_module, backward=False)

        see_memory_usage(
            f"After sub module function {sub_module.__class__.__name__}  {sub_module.id} after release",
            force=False,
        )

    @torch.no_grad()
    def pre_sub_module_backward_function(self, sub_module):
        assert (
            sub_module.training
        ), "backward pass is invalid for module in evaluation mode"
        param_coordinator = self.get_param_coordinator(training=True)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module, forward=False)

    @torch.no_grad()
    def post_sub_module_backward_function(self, sub_module):
        assert (
            sub_module.training
        ), "backward pass is invalid for module in evaluation mode"
        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} before release",
            force=False,
        )

        self.get_param_coordinator(training=True).release_sub_module(
            sub_module, backward=True
        )

        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} after release",
            force=False,
        )
