#!/usr/bin/env python3
import threading
import typing
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from explainability.integrated_gradients._utils.common import (
    _reduce_list,
    _run_forward,
    _sort_key_list,
    _verify_select_neuron,
)
from explainability.integrated_gradients._utils.typing import (
    Literal,
    ModuleOrModuleList,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torch import Tensor, device
from torch.nn import Module


def apply_gradient_requirements(
        inputs: Tuple[Tensor, ...], warn: bool = True
) -> List[bool]:
    """
    Iterates through tuple on input tensors and sets requires_grad to be true on
    each Tensor, and ensures all grads are set to zero. To ensure that the input
    is returned to its initial state, a list of flags representing whether or not
     a tensor originally required grad is returned.
    """
    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients"
    grad_required = []
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        grad_required.append(input.requires_grad)
        inputs_dtype = input.dtype
        # Note: torch 1.2 doesn't support is_complex for dtype that's why we check
        # on the existance of is_complex method.
        if not inputs_dtype.is_floating_point and not (
                hasattr(inputs_dtype, "is_complex") and inputs_dtype.is_complex
        ):
            if warn:
                warnings.warn(
                    """Input Tensor %d has a dtype of %s.
                    Gradients cannot be activated
                    for these data types."""
                    % (index, str(inputs_dtype))
                )
        elif not input.requires_grad:
            if warn:
                warnings.warn(
                    "Input Tensor %d did not already require gradients, "
                    "required_grads has been set automatically." % index
                )
            input.requires_grad_()
    return grad_required


def undo_gradient_requirements(
        inputs: Tuple[Tensor, ...], grad_required: List[bool]
) -> None:
    """
    Iterates through list of tensors, zeros each gradient, and sets required
    grad to false if the corresponding index in grad_required is False.
    This method is used to undo the effects of prepare_gradient_inputs, making
    grads not required for any input tensor that did not initially require
    gradients.
    """

    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients."
    assert len(inputs) == len(
        grad_required
    ), "Input tuple length should match gradient mask."
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        if not grad_required[index]:
            input.requires_grad_(False)


def compute_gradients(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(outputs), inputs)
    return grads


def _neuron_gradients(
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        saved_layer: Dict[device, Tuple[Tensor, ...]],
        key_list: List[device],
        gradient_neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
) -> Tuple[Tensor, ...]:
    with torch.autograd.set_grad_enabled(True):
        gradient_tensors = []
        for key in key_list:
            current_out_tensor = _verify_select_neuron(
                saved_layer[key], gradient_neuron_selector
            )
            gradient_tensors.append(
                torch.autograd.grad(
                    torch.unbind(current_out_tensor)
                    if current_out_tensor.numel() > 1
                    else current_out_tensor,
                    inputs,
                )
            )
        _total_gradients = _reduce_list(gradient_tensors, sum)
    return _total_gradients


@typing.overload
def _forward_layer_eval(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: Module,
        additional_forward_args: Any = None,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
        grad_enabled: bool = False,
) -> Tuple[Tensor, ...]:
    ...


@typing.overload
def _forward_layer_eval(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: List[Module],
        additional_forward_args: Any = None,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
        grad_enabled: bool = False,
) -> List[Tuple[Tensor, ...]]:
    ...


def _forward_layer_eval(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: ModuleOrModuleList,
        additional_forward_args: Any = None,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
        grad_enabled: bool = False,
) -> Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]:
    return _forward_layer_eval_with_neuron_grads(
        forward_fn,
        inputs,
        layer,
        additional_forward_args=additional_forward_args,
        gradient_neuron_selector=None,
        grad_enabled=grad_enabled,
        device_ids=device_ids,
        attribute_to_layer_input=attribute_to_layer_input,
    )


@typing.overload
def _forward_layer_distributed_eval(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: ModuleOrModuleList,
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        forward_hook_with_return: Literal[False] = False,
        require_layer_grads: bool = False,
) -> Dict[Module, Dict[device, Tuple[Tensor, ...]]]:
    ...


@typing.overload
def _forward_layer_distributed_eval(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: ModuleOrModuleList,
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        *,
        forward_hook_with_return: Literal[True],
        require_layer_grads: bool = False,
) -> Tuple[Dict[Module, Dict[device, Tuple[Tensor, ...]]], Tensor]:
    ...


def _forward_layer_distributed_eval(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: ModuleOrModuleList,
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        forward_hook_with_return: bool = False,
        require_layer_grads: bool = False,
) -> Union[
    Tuple[Dict[Module, Dict[device, Tuple[Tensor, ...]]], Tensor],
    Dict[Module, Dict[device, Tuple[Tensor, ...]]],
]:
    r"""
    A helper function that allows to set a hook on model's `layer`, run the forward
    pass and returns intermediate layer results, stored in a dictionary,
    and optionally also the output of the forward function. The keys in the
    dictionary are the device ids and the values are corresponding intermediate layer
    results, either the inputs or the outputs of the layer depending on whether we set
    `attribute_to_layer_input` to True or False.
    This is especially useful when we execute forward pass in a distributed setting,
    using `DataParallel`s for example.
    """
    saved_layer: Dict[Module, Dict[device, Tuple[Tensor, ...]]] = defaultdict(dict)
    lock = threading.Lock()
    all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer

    # Set a forward hook on specified module and run forward pass to
    # get layer output tensor(s).
    # For DataParallel models, each partition adds entry to dictionary
    # with key as device and value as corresponding Tensor.
    def hook_wrapper(original_module):
        def forward_hook(module, inp, out=None):
            eval_tsrs = inp if attribute_to_layer_input else out
            is_eval_tuple = isinstance(eval_tsrs, tuple)

            if not is_eval_tuple:
                eval_tsrs = (eval_tsrs,)
            if require_layer_grads:
                apply_gradient_requirements(eval_tsrs, warn=False)
            with lock:
                nonlocal saved_layer
                # Note that cloning behaviour of `eval_tsr` is different
                # when `forward_hook_with_return` is set to True. This is because
                # otherwise `backward()` on the last output layer won't execute.
                if forward_hook_with_return:
                    saved_layer[original_module][eval_tsrs[0].device] = eval_tsrs
                    eval_tsrs_to_return = tuple(
                        eval_tsr.clone() for eval_tsr in eval_tsrs
                    )
                    if not is_eval_tuple:
                        eval_tsrs_to_return = eval_tsrs_to_return[0]
                    return eval_tsrs_to_return
                else:
                    saved_layer[original_module][eval_tsrs[0].device] = tuple(
                        eval_tsr.clone() for eval_tsr in eval_tsrs
                    )

        return forward_hook

    all_hooks = []
    try:
        for single_layer in all_layers:
            if attribute_to_layer_input:
                all_hooks.append(
                    single_layer.register_forward_pre_hook(hook_wrapper(single_layer))
                )
            else:
                all_hooks.append(
                    single_layer.register_forward_hook(hook_wrapper(single_layer))
                )
        output = _run_forward(
            forward_fn,
            inputs,
            target=target_ind,
            additional_forward_args=additional_forward_args,
        )
    finally:
        for hook in all_hooks:
            hook.remove()

    if len(saved_layer) == 0:
        raise AssertionError("Forward hook did not obtain any outputs for given layer")

    if forward_hook_with_return:
        return saved_layer, output
    return saved_layer


def _gather_distributed_tensors(
        saved_layer: Dict[device, Tuple[Tensor, ...]],
        device_ids: Union[None, List[int]] = None,
        key_list: Union[None, List[device]] = None,
) -> Tuple[Tensor, ...]:
    r"""
    A helper function to concatenate intermediate layer results stored on
    different devices in `saved_layer`. `saved_layer` is a dictionary that
    contains `device_id` as a key and intermediate layer results (either
    the input or the output of the layer) stored on the device corresponding to
    the key.
    `key_list` is a list of devices in appropriate ordering for concatenation
    and if not provided, keys are sorted based on device ids.

    If only one key exists (standard model), key list simply has one element.
    """
    if key_list is None:
        key_list = _sort_key_list(list(saved_layer.keys()), device_ids)
    return _reduce_list([saved_layer[device_id] for device_id in key_list])


def _extract_device_ids(
        forward_fn: Callable,
        saved_layer: Dict[Module, Dict[device, Tuple[Tensor, ...]]],
        device_ids: Union[None, List[int]],
) -> Union[None, List[int]]:
    r"""
    A helper function to extract device_ids from `forward_function` in case it is
    provided as part of a `DataParallel` model or if is accessible from
    `forward_fn`.
    In case input device_ids is not None, this function returns that value.
    """
    # Multiple devices / keys implies a DataParallel model, so we look for
    # device IDs if given or available from forward function
    # (DataParallel model object).
    if (
            max(len(saved_layer[single_layer]) for single_layer in saved_layer) > 1
            and device_ids is None
    ):
        if (
                hasattr(forward_fn, "device_ids")
                and cast(Any, forward_fn).device_ids is not None
        ):
            device_ids = cast(Any, forward_fn).device_ids
        else:
            raise AssertionError(
                "Layer tensors are saved on multiple devices, however unable to access"
                " device ID list from the `forward_fn`. Device ID list must be"
                " accessible from `forward_fn`. For example, they can be retrieved"
                " if `forward_fn` is a model of type `DataParallel`. It is used"
                " for identifying device batch ordering."
            )
    return device_ids


@typing.overload
def _forward_layer_eval_with_neuron_grads(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: Module,
        additional_forward_args: Any = None,
        *,
        gradient_neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
        grad_enabled: bool = False,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    ...


@typing.overload
def _forward_layer_eval_with_neuron_grads(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: Module,
        additional_forward_args: Any = None,
        gradient_neuron_selector: None = None,
        grad_enabled: bool = False,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
) -> Tuple[Tensor, ...]:
    ...


@typing.overload
def _forward_layer_eval_with_neuron_grads(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: List[Module],
        additional_forward_args: Any = None,
        gradient_neuron_selector: None = None,
        grad_enabled: bool = False,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
) -> List[Tuple[Tensor, ...]]:
    ...


def _forward_layer_eval_with_neuron_grads(
        forward_fn: Callable,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer: ModuleOrModuleList,
        additional_forward_args: Any = None,
        gradient_neuron_selector: Union[
            None, int, Tuple[Union[int, slice], ...], Callable
        ] = None,
        grad_enabled: bool = False,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tensor, ...],
    List[Tuple[Tensor, ...]],
]:
    """
    This method computes forward evaluation for a particular layer using a
    forward hook. If a gradient_neuron_selector is provided, then gradients with
    respect to that neuron in the layer output are also returned.

    These functionalities are combined due to the behavior of DataParallel models
    with hooks, in which hooks are executed once per device. We need to internally
    combine the separated tensors from devices by concatenating based on device_ids.
    Any necessary gradients must be taken with respect to each independent batched
    tensor, so the gradients are computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel models
    can be found in the PyTorch data parallel documentation. We maintain the separate
    evals in a dictionary protected by a lock, analogous to the gather implementation
    for the core PyTorch DataParallel implementation.
    """
    grad_enabled = True if gradient_neuron_selector is not None else grad_enabled

    with torch.autograd.set_grad_enabled(grad_enabled):
        saved_layer = _forward_layer_distributed_eval(
            forward_fn,
            inputs,
            layer,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
    device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)
    # Identifies correct device ordering based on device ids.
    # key_list is a list of devices in appropriate ordering for concatenation.
    # If only one key exists (standard model), key list simply has one element.
    key_list = _sort_key_list(list(next(iter(saved_layer.values())).keys()), device_ids)
    if gradient_neuron_selector is not None:
        assert isinstance(
            layer, Module
        ), "Cannot compute neuron gradients for multiple layers simultaneously!"
        inp_grads = _neuron_gradients(
            inputs, saved_layer[layer], key_list, gradient_neuron_selector
        )
        return (
            _gather_distributed_tensors(saved_layer[layer], key_list=key_list),
            inp_grads,
        )
    else:
        if isinstance(layer, Module):
            return _gather_distributed_tensors(saved_layer[layer], key_list=key_list)
        else:
            return [
                _gather_distributed_tensors(saved_layer[curr_layer], key_list=key_list)
                for curr_layer in layer
            ]


@typing.overload
def compute_layer_gradients_and_eval(
        forward_fn: Callable,
        layer: Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
        *,
        gradient_neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
        output_fn: Union[None, Callable] = None,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    ...


@typing.overload
def compute_layer_gradients_and_eval(
        forward_fn: Callable,
        layer: List[Module],
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
        gradient_neuron_selector: None = None,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
        output_fn: Union[None, Callable] = None,
) -> Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]]:
    ...


@typing.overload
def compute_layer_gradients_and_eval(
        forward_fn: Callable,
        layer: Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
        gradient_neuron_selector: None = None,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
        output_fn: Union[None, Callable] = None,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    ...


def compute_layer_gradients_and_eval(
        forward_fn: Callable,
        layer: ModuleOrModuleList,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
        gradient_neuron_selector: Union[
            None, int, Tuple[Union[int, slice], ...], Callable
        ] = None,
        device_ids: Union[None, List[int]] = None,
        attribute_to_layer_input: bool = False,
        output_fn: Union[None, Callable] = None,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]],
]:
    r"""
    Computes gradients of the output with respect to a given layer as well
    as the output evaluation of the layer for an arbitrary forward function
    and given input.

    For data parallel models, hooks are executed once per device ,so we
    need to internally combine the separated tensors from devices by
    concatenating based on device_ids. Any necessary gradients must be taken
    with respect to each independent batched tensor, so the gradients are
    computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel
    models can be found in the PyTorch data parallel documentation. We maintain
    the separate inputs in a dictionary protected by a lock, analogous to the
    gather implementation for the core PyTorch DataParallel implementation.

    NOTE: To properly handle inplace operations, a clone of the layer output
    is stored. This structure inhibits execution of a backward hook on the last
    module for the layer output when computing the gradient with respect to
    the input, since we store an intermediate clone, as
    opposed to the true module output. If backward module hooks are necessary
    for the final module when computing input gradients, utilize
    _forward_layer_eval_with_neuron_grads instead.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        layer:      Layer for which gradients / output will be evaluated.
        inputs:     Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        output_fn:  An optional function that is applied to the layer inputs or
                    outputs depending whether the `attribute_to_layer_input` is
                    set to `True` or `False`
        args:       Additional input arguments that forward function requires.
                    It takes an empty tuple (no additional arguments) if no
                    additional arguments are required


    Returns:
        2-element tuple of **gradients**, **evals**:
        - **gradients**:
            Gradients of output with respect to target layer output.
        - **evals**:
            Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        # saved_layer is a dictionary mapping device to a tuple of
        # layer evaluations on that device.
        saved_layer, output = _forward_layer_distributed_eval(
            forward_fn,
            inputs,
            layer,
            target_ind=target_ind,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            forward_hook_with_return=True,
            require_layer_grads=True,
        )
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(
            list(next(iter(saved_layer.values())).keys()), device_ids
        )
        all_outputs: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        if isinstance(layer, Module):
            all_outputs = _reduce_list(
                [
                    saved_layer[layer][device_id]
                    if output_fn is None
                    else output_fn(saved_layer[layer][device_id])
                    for device_id in key_list
                ]
            )
        else:
            all_outputs = [
                _reduce_list(
                    [
                        saved_layer[single_layer][device_id]
                        if output_fn is None
                        else output_fn(saved_layer[single_layer][device_id])
                        for device_id in key_list
                    ]
                )
                for single_layer in layer
            ]
        all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer
        grad_inputs = tuple(
            layer_tensor
            for single_layer in all_layers
            for device_id in key_list
            for layer_tensor in saved_layer[single_layer][device_id]
        )
        saved_grads = torch.autograd.grad(torch.unbind(output), grad_inputs)

        offset = 0
        all_grads: List[Tuple[Tensor, ...]] = []
        for single_layer in all_layers:
            num_tensors = len(next(iter(saved_layer[single_layer].values())))
            curr_saved_grads = [
                saved_grads[i : i + num_tensors]
                for i in range(
                    offset, offset + len(key_list) * num_tensors, num_tensors
                )
            ]
            offset += len(key_list) * num_tensors
            if output_fn is not None:
                curr_saved_grads = [
                    output_fn(curr_saved_grad) for curr_saved_grad in curr_saved_grads
                ]

            all_grads.append(_reduce_list(curr_saved_grads))

        layer_grads: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        layer_grads = all_grads
        if isinstance(layer, Module):
            layer_grads = all_grads[0]

        if gradient_neuron_selector is not None:
            assert isinstance(
                layer, Module
            ), "Cannot compute neuron gradients for multiple layers simultaneously!"
            inp_grads = _neuron_gradients(
                inputs, saved_layer[layer], key_list, gradient_neuron_selector
            )
            return (
                cast(Tuple[Tensor, ...], layer_grads),
                cast(Tuple[Tensor, ...], all_outputs),
                inp_grads,
            )
    return layer_grads, all_outputs  # type: ignore


def construct_neuron_grad_fn(
        layer: Module,
        neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
        device_ids: Union[None, List[int]] = None,
        attribute_to_neuron_input: bool = False,
) -> Callable:
    def grad_fn(
            forward_fn: Callable,
            inputs: TensorOrTupleOfTensorsGeneric,
            target_ind: TargetType = None,
            additional_forward_args: Any = None,
    ) -> Tuple[Tensor, ...]:
        _, grads = _forward_layer_eval_with_neuron_grads(
            forward_fn,
            inputs,
            layer,
            additional_forward_args,
            gradient_neuron_selector=neuron_selector,
            device_ids=device_ids,
            attribute_to_layer_input=attribute_to_neuron_input,
        )
        return grads

    return grad_fn


def _compute_jacobian_wrt_params(
        model: Module,
        inputs: Union[Tuple[Tensor], Tensor],
        labels: Optional[Tensor] = None,
        loss_fn: Optional[Union[Module, Callable]] = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes the Jacobian of a batch of test examples given a model, and optional
    loss function and target labels. This method is equivalent to calculating the
    gradient for every individual example in the minibatch.

    Args:
        model (torch.nn.Module): The trainable model providing the forward pass
        inputs (Tensor): The minibatch for which the forward pass is computed.
                The dimensions of input are (N, *) where N is the batch_size.
                The input must have a batch dimension, even if batch_size = 1.
        labels (Tensor or None): Labels for input if computing a loss function.
        loss_fn (torch.nn.Module or Callable or None): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='none'`.

    Returns:
        grads (Tuple of Tensor): Returns the Jacobian for the minibatch as a
                tuple of gradients corresponding to the tuple of trainable parameters
                returned by `model.parameters()`. Each object grads[i] references to the
                gradients for the parameters in the i-th trainable layer of the model.
                Each grads[i] object is a tensor with the gradients for the `inputs`
                batch. For example, grads[i][j] would reference the gradients for the
                parameters of the i-th layer, for the j-th member of the minibatch.
    """
    with torch.autograd.set_grad_enabled(True):
        out = model(inputs)
        assert out.dim() != 0, "Please ensure model output has at least one dimension."

        if labels is not None and loss_fn is not None:
            loss = loss_fn(out, labels)
            if hasattr(loss_fn, "reduction"):
                msg0 = "Please ensure loss_fn.reduction is set to `none`"
                assert loss_fn.reduction == "none", msg0  # type: ignore
            else:
                msg1 = (
                    "Loss function is applying a reduction. Please ensure "
                    f"Output shape: {out.shape} and Loss shape: {loss.shape} "
                    "are matching."
                )
                assert loss.dim() != 0, msg1
                assert out.shape[0] == loss.shape[0], msg1
            out = loss

        grads_list = [
            torch.autograd.grad(
                outputs=out[i],
                inputs=model.parameters(),  # type: ignore
                grad_outputs=torch.ones_like(out[i]),
                retain_graph=True,
            )
            for i in range(out.shape[0])
        ]

        grads = tuple([torch.stack(x) for x in zip(*grads_list)])

        return tuple(grads)


def _compute_jacobian_wrt_params_autograd_hacks(
        model: Module,
        inputs: Union[Tuple[Tensor], Tensor],
        labels: Optional[Tensor] = None,
        loss_fn: Optional[Module] = None,
        reduction_type: Optional[str] = "sum",
) -> Tuple[Any, ...]:
    r"""
    NOT SUPPORTED FOR OPEN SOURCE. This method uses an internal 'hack` and is currently
    not supported.

    Computes the Jacobian of a batch of test examples given a model, and optional
    loss function and target labels. This method uses autograd_hacks to fully vectorize
    the Jacobian calculation. Currently, only linear and conv2d layers are supported.

    User must `add_hooks(model)` before calling this function.

    Args:
        model (torch.nn.Module): The trainable model providing the forward pass
        inputs (Tensor): The minibatch for which the forward pass is computed.
                The dimensions of input are (N, *) where N is the batch_size.
                The input must have a batch dimension, even if batch_size = 1.
        labels (Tensor or None): Labels for input if computing a loss function.
        loss_fn (torch.nn.Module or Callable or None): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='sum'` or
                `reduction='mean'`.
        reduction_type (str): The type of reduction applied. If a loss_fn is passed,
                this should match `loss_fn.reduction`. Else if gradients are being
                computed on direct model outputs (scores), then 'sum' should be used.
                Defaults to 'sum'.

    Returns:
        grads (Tuple of Tensor): Returns the Jacobian for the minibatch as a
                tuple of gradients corresponding to the tuple of trainable parameters
                returned by `model.parameters()`. Each object grads[i] references to the
                gradients for the parameters in the i-th trainable layer of the model.
                Each grads[i] object is a tensor with the gradients for the `inputs`
                batch. For example, grads[i][j] would reference the gradients for the
                parameters of the i-th layer, for the j-th member of the minibatch.
    """
    from captum._utils.fb import autograd_hacks

    with torch.autograd.set_grad_enabled(True):
        autograd_hacks.add_hooks(model)

        out = model(inputs)
        assert out.dim() != 0, "Please ensure model output has at least one dimension."

        if labels is not None and loss_fn is not None:
            loss = loss_fn(out, labels)
            if hasattr(loss_fn, "reduction"):
                msg0 = "Please ensure loss_fn.reduction is set to `sum` or `mean`"
                assert loss_fn.reduction != "none", msg0
                msg1 = (
                    f"loss_fn.reduction ({loss_fn.reduction}) does not match reduction "
                    f"type ({reduction_type}). Please ensure they are matching."
                )
                assert loss_fn.reduction == reduction_type, msg1
            msg2 = (
                "Please ensure custom loss function is applying either a "
                "sum or mean reduction."
            )
            assert out.shape != loss.shape, msg2

            if reduction_type != "sum" and reduction_type != "mean":
                raise ValueError(
                    f"{reduction_type} is not a valid value for reduction_type. "
                    "Must be either 'sum' or 'mean'."
                )
            out = loss

        model.zero_grad()
        out.backward(gradient=torch.ones_like(out))
        autograd_hacks.compute_grad1(model, loss_type=reduction_type)

        grads = tuple(
            param.grad1  # type: ignore
            for param in model.parameters()
            if hasattr(param, "grad1")
        )

        autograd_hacks.clear_backprops(model)
        autograd_hacks.remove_hooks(model)

        return grads
