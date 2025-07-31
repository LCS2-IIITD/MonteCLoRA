# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
import logging
import os
import re
import textwrap
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Union
import sys

import torch
from accelerate import init_empty_weights
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.wishart import Wishart
from torch.distributions.gamma import Gamma


import torch.nn.functional as F
import threading
import queue
import time

from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from peft.utils.constants import (
    DUMMY_MODEL_CONFIG,
    DUMMY_TARGET_MODULES,
    EMBEDDING_LAYER_NAMES,
    MIN_TARGET_MODULES_FOR_OPTIMIZATION,
    SEQ_CLS_HEAD_NAMES,
)
from peft.utils.peft_types import PeftType, TaskType

from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
from ._buffer_dict import BufferDict



logger = logging.getLogger(__name__)


@contextmanager
def onload_layer(layer):
    r"""
    A utility for modifying a module containing one or more tuners and a base layer, any of which are offloaded to the
    CPU or disk. Moves a module's sub-modules to the execution device before some action is performed, after that the
    base layer state dictionary is re-assigned (if that layer was offloaded to the disk) and finally the parameters are
    offloaded.

    If the module has no offloaded sub-modules, this function does nothing.

    Args:
        layer ('torch.nn.Module'):
            layer with tuners to be merged
    """

    offloaded_modules = []
    for name, module in layer.named_modules():
        if name in ["", "base_layer"]:
            continue
        if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            module._hf_hook.pre_forward(module)
            offloaded_modules.append(module)

    base_layer_offload = False
    if hasattr(layer, "base_layer") and (
        hasattr(layer.base_layer, "_hf_hook")
        and isinstance(layer.base_layer._hf_hook, AlignDevicesHook)
        and layer.base_layer._hf_hook.offload
    ):
        # check if the base layer is disk-offloaded (must contain a 'dataset' and an offload index)
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # find the disk-offload index (maps modules to safetensors) from the `dataset` (OffloadedWeightsLoader object)
            index = layer.base_layer._hf_hook.weights_map.dataset.index
            module_name = list(dict(layer.base_layer._hf_hook.weights_map.dataset).keys())[0]  # any module will do
            file_name = index[module_name]["safetensors_file"]
            base_name_arr = []
            # get effective dir name
            for i in os.path.split(file_name):
                if "--" in i:
                    base_name_arr.append(i)
                    break
                base_name_arr.append(i)
            base_name = os.path.join(*base_name_arr)
            safetensors_filename = base_name + "-merged"
        layer.base_layer._hf_hook.pre_forward(layer.base_layer)
        base_layer_offload = True

    yield

    for module in offloaded_modules:
        module._hf_hook.post_forward(module, torch.tensor([]))

    if base_layer_offload:
        # re-make weights map (must be on cpu to send params to the disk via memmap if disk offload)
        layer.base_layer._hf_hook.weights_map = {
            name: param.to("cpu") for name, param in named_module_tensors(layer.base_layer)
        }
        # offload weights map to disk if original device is the disk
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # rewrite directory with merged weights
            offload_state_dict(safetensors_filename, layer.base_layer._hf_hook.weights_map)
        layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))


class BaseTuner(nn.Module, ABC):
    r"""
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_create_and_replace**:
        A private method to create and replace the target module with the adapter module.
    - **_check_target_module_exists**:
        A private helper method to check if the passed module's key name matches any of the target modules in the
        adapter_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `PeftConfig` objects. One can also
            pass a PeftConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
        targeted_module_names (`list[str]`):
            The list of module names that were actually adapted. Can be useful to inspect if you want to quickly
            double-check that the `config.target_modules` were specified correctly.
    """

    def __init__(
        self,
        model,
        peft_config: Union[PeftConfig, dict[str, PeftConfig]],
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        super().__init__()

        self.model = model
        self.targeted_module_names: list[str] = []

        # For advanced developers, if you want to attach multiple adapters to your
        # model, just add a `peft_config` dict attribute to your model.
        if not hasattr(self, "peft_config"):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            logger.info(
                "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters"
                " in the model. Make sure to know what you are doing!"
            )
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                # user is adding a dict of PeftConfigs
                self.peft_config.update(peft_config)

        self.active_adapter: str | list[str] = adapter_name
        self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)
        if peft_config != PeftType.XLORA or peft_config[adapter_name] != PeftType.XLORA:
            self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        # Copy the peft_config in the injected model.
        self.model.peft_config = self.peft_config

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def _pre_injection_hook(self, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        r"""
        A hook to be called before the adapter is injected into the model. This method can be overridden by child
        classes to perform any pre-injection operations.

        Args:
            model (`nn.Module`):
                The model to be adapted.
            config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
        """
        pass

    @abstractmethod
    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        r"""
        A private method to eventually prepare the adapter config. For transformers based models, if
        `peft_config.target_modules` is None, we can automatically infer the target modules from the
        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to
        automatically infer it for all tuner models.

        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            model_config (`dict`):
                The transformers model config, that config should contain the `model_type` key.
        """
        ...

    def _prepare_model(self, peft_config: PeftConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        See `peft.tuner.lora.LoraModel._prepare_model` for an example.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        pass

    @abstractmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        r"""
        A helper private method to check if the passed module's key name matches any of the target modules in the
        `peft_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        ...

    @abstractmethod
    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overridden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            current_key (`str`):
                The key of the current target being adapted.
        """
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        r"""
        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False) This needs to
        be overridden for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.
        """
        ...

    @abstractmethod
    def disable_adapter_layers(self) -> None:
        """
        Disable all adapters in-place.
        """
        ...

    @abstractmethod
    def enable_adapter_layers(self) -> None:
        """
        Enable all adapters in-place
        """
        ...

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        pass

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        """
        A helper method to cast the adapter weights to the correct dtype.

        Currently, this only upcasts float16 and bfloat16 to float32.

        Args:
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.

        """
        if not autocast_adapter_dtype:
            return

        dtypes_to_convert_to_fp32 = {torch.float16, torch.bfloat16}

        for module in self.model.modules():
            if not isinstance(module, BaseTunerLayer):
                continue

            for submodule in module.modules():
                if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                    continue

                if adapter_name not in submodule:
                    continue

                if isinstance(submodule[adapter_name], nn.Parameter):
                    if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                        submodule[adapter_name].data = submodule[adapter_name].data.to(torch.float32)
                    continue

                if isinstance(submodule[adapter_name], torch.Tensor):  # e.g. from a BufferDict
                    if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                        submodule[adapter_name] = submodule[adapter_name].to(torch.float32)
                    continue

                for param in submodule[adapter_name].parameters():
                    if param.dtype in dtypes_to_convert_to_fp32:
                        param.data = param.data.to(torch.float32)

    def _check_merge_allowed(self):
        """Helper method to check whether the adapter can be merged.

        Raise a ValueError if it is not possible to merge the adapter with the given configuration.
        """
        example_code = textwrap.dedent(
            """
            ```python
            from transformers import AutoModelForCausalLM

            # Load original tied model
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", tie_word_embeddings=False)

            # Set the randomly initialized lm_head to the previously tied embeddings
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

            # Save the untied model
            untied_model_dir = "dir/for/untied/model"
            model.save_pretrained(untied_model_dir)
            model.config.save_pretrained(untied_model_dir)

            # Now use the original model but in untied format
            model = AutoModelForCausalLM.from_pretrained(untied_model_dir)
            ```
            """
        )
        tied_target_modules = self._get_tied_target_modules(self.model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
                "This can lead to complications. "
                "You can opt to merge the adapter after cloning the weights (to untie the embeddings). "
                "You can untie the embeddings by loading the model with `tie_word_embeddings=False`. For example:"
                + example_code
            )

    def inject_adapter(
        self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False
    ) -> None:
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the loading process.

        """
        peft_config = self.peft_config[adapter_name]
        excluded_modules = []
        unmatched_modules = []
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False

        model_config = self.get_model_config(model)

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)
        key_list = [key for key, _ in model.named_modules()]

        uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        if uses_dummy_target_modules:
            # dummy adapter, we allow not matching any module
            key_list = []

        # update peft_config.target_modules if required
        peft_config = _maybe_include_all_linear_layers(peft_config, model)

        # This is an optimization to reduce the number of entries in the target_modules list. The reason is that in some
        # circumstances, target_modules can contain hundreds of entries. Since each target module is checked against
        # each module of the net (which can be thousands), this can become quite expensive when many adapters are being
        # added. Often, the target_modules can be condensed in such a case, which speeds up the process.
        # A context in which this can happen is when diffusers loads non-PEFT LoRAs. As there is no meta info on
        # target_modules in that case, they are just inferred by listing all keys from the state_dict, which can be
        # quite a lot. See: https://github.com/huggingface/diffusers/issues/9297
        # As there is a small chance for undiscovered bugs, we apply this optimization only if the list of
        # target_modules is sufficiently big.
        if (
            isinstance(peft_config.target_modules, (list, set))
            and len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION
        ):
            names_no_target = [
                name for name in key_list if not any(name.endswith(suffix) for suffix in peft_config.target_modules)
            ]
            new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
            if len(new_target_modules) < len(peft_config.target_modules):
                peft_config.target_modules = new_target_modules

        for key in key_list:
            if not key:
                continue
            # Check for modules_to_save in case
            if _check_for_modules_to_save and any(
                key.endswith(f"{module_to_save}") for module_to_save in peft_config.modules_to_save
            ):
                # Optionally set the modules to save
                parent, target, target_name = _get_submodules(model, key)

                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True
                continue

            result = self._check_target_module_exists(peft_config, key)
            if isinstance(result, _ExcludedModule):
                excluded_modules.append(key)
            elif not result:
                unmatched_modules.append(key)
            else:
                self.targeted_module_names.append(key)
                parent, target, target_name = _get_submodules(model, key)
                ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                with ctx():
                    self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)

        if not self.targeted_module_names and not uses_dummy_target_modules:
            if excluded_modules and not unmatched_modules:
                # All targeted modules were excluded
                raise ValueError(
                    "All modules were excluded. This is likely unintended. "
                    "Check your `target_modules` and `exclude_modules` configuration."
                )
            elif not excluded_modules and unmatched_modules:
                # None of the targeted modules matched
                raise ValueError(
                    f"Target modules {peft_config.target_modules} not found in the base model. "
                    f"Please check the target modules and try again."
                )
            else:
                # Some modules did not match and some matched but were excluded
                raise ValueError(
                    "No modules were targeted for adaptation. "
                    "This might be caused by a combination of mismatched target modules and excluded modules. "
                    "Please check your `target_modules` and `exclude_modules` configuration."
                )

        elif hasattr(peft_config, "exclude_modules") and peft_config.exclude_modules and not excluded_modules:
            # exclude_modules was passed but was not used
            warnings.warn(
                f"You have passed exclude_modules={peft_config.exclude_modules} but no modules were excluded. "
                "Please check that exclude_modules was set correctly."
            )

        tied_target_modules = self._get_tied_target_modules(model=model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
                "This can lead to complications, for example when merging the adapter "
                "or converting your model to formats other than safetensors. "
                "See for example https://github.com/huggingface/peft/issues/2018."
            )

        # It's important to set the adapter here (again), because otherwise it can happen that if a 2nd adapter is
        # added, and it targets different layer(s) than the first adapter (which is active), then those different
        # layers will be activated, which we don't want.
        self.set_adapter(self.active_adapters)
        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, "modules_to_save"):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def merge_adapter(self, adapter_names: Optional[list[str]] = None) -> None:
        """
        This method merges the adapter layers into the base model.

        Merging adapters can lead to a speed up of the forward pass. A copy of the adapter weights is still kept in
        memory, which is required to unmerge the adapters. In order to merge the adapter weights without keeping them
        in memory, please call `merge_and_unload`.

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        self._check_merge_allowed()
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.merge(adapter_names=adapter_names)

    def unmerge_adapter(self):
        """
        This method unmerges all merged adapter layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.unmerge()

    def _unloading_checks(self, adapter_names: Optional[list[str]]):
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any(
            self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider
        )
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError("Cannot unload multiple adapters that specify `modules_to_save`.")

    @staticmethod
    def get_model_config(model: nn.Module) -> dict:
        """
        This method gets the config from a model in dictionary form. If model has not attribute config, then this
        method returns a default config.

        Args:
            model (`nn.Module`):
                Model to get the config from.
            default (`dict|None`, *optional*)::
                What to return if model does not have a config attribute.
        """
        model_config = getattr(model, "config", DUMMY_MODEL_CONFIG)
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        return model_config

    def _get_tied_target_modules(self, model: nn.Module) -> list[str]:
        tied_target_modules = []
        model_config = self.get_model_config(model)
        if model_config.get("tie_word_embeddings"):
            for target_module in self.targeted_module_names:
                if target_module in EMBEDDING_LAYER_NAMES:
                    tied_target_modules.append(target_module)
        return tied_target_modules


class BaseTunerLayer(ABC):
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_pluggable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """

    # All names of layers that may contain adapter (trainable) weights
    adapter_layer_names: tuple[str, ...] = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ()

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: str | list[str] = "default"

    # List all merged adapters
    merged_adapters: list[str] = []

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    @property
    def weight(self) -> torch.Tensor:
        # This is required for some transformers code, e.g. for T5, weight is accessed as:
        #     self.wo.weight
        # where "wo" is the adapter layer.
        # https://github.com/huggingface/transformers/blob/78f6ed6c70b29c1560780e3869a7ad4c6b3d2710/src/transformers
        # /models/t5/modeling_t5.py#L292
        base_layer = self.get_base_layer()
        if hasattr(base_layer, "qweight"):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight

    @property
    def bias(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return base_layer.bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str | list[str]:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    def _get_available_adapters(self) -> set[str]:
        """Return all adapter names that can be found on this module."""
        adapters = set()
        for layer_name in self.adapter_layer_names:
            module = getattr(self, layer_name)
            if not isinstance(module, (nn.ModuleDict, nn.ParameterDict)):
                continue
            adapters.update(set(module.keys()))
        return adapters

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list[str]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        if device is None:
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(self.get_base_layer(), weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        meta = torch.device("meta")

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue

            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

eps = 1e-3

# class AsyncMonteCLoRASampler(threading.Thread):
#     def __init__(self, model, buffer_size=10, device='cpu'):
#         super().__init__(daemon=True)
#         self.model = model
#         self.device = device
#         self.buffer_size = 10
#         self.queue = queue.Queue(maxsize=buffer_size)
#         self.running = True

#     def run(self):
#         while self.running:
#             if self.queue.qsize() < self.buffer_size:
#                 # print(self.queue.qsize())
#                 try:
#                     with torch.no_grad():
#                         z_mvn = torch.randn(
#                             (self.model.monteclora_n, self.model.in_features, self.model.out_features),
#                             device=self.device
#                         )
#                         wishart_sampler = Wishart(
#                             df=self.model.out_features,
#                             scale_tril=torch.eye(self.model.out_features, device=self.device)
#                         )
#                         z_wishart = wishart_sampler._bartlett_sampling(torch.Size())

#                         z_dirichlet = torch.randn(self.model.monteclora_n, device=self.device)

#                         sample = {
#                             'z_mvn': z_mvn,
#                             'z_wishart': z_wishart,
#                             'z_dirichlet': z_dirichlet
#                         }

#                         self.queue.put_nowait(sample)
#                 except KeyboardInterrupt:
#                     sys.exit(0)
#                 except Exception as e:
#                     print(f"[AsyncMonteCLoRASampler] Error: {e}")
#             else:
#                 time.sleep(0.01)

#     def get(self):
#         try:
#             return self.queue.get_nowait()
#         except queue.Empty:
#             return None

#     def stop(self):
#         self.running = False
#         self.join(timeout=1.0)

# class MonteCLoRASampler(nn.Module):
#     def __init__(
#         self, 
#         in_features, 
#         out_features, 
#         monteclora_n,
#         fan_in_fan_out=False,
#         use_entropy=True,
#         dirichlet_prior=1,
#         sample_scaler=3e-4,
#         kl_loss_weight=1e-5,
#         mc_training=True,
#         buffer_size=10,
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         monteclora_m = None
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.monteclora_n = monteclora_n
#         self.fan_in_fan_out = fan_in_fan_out
#         self.use_entropy = use_entropy
#         self.device = device
#         self.sample_scaler = sample_scaler
#         self.kl_loss_weight = kl_loss_weight
#         self.mc_training = mc_training
#         self.dirichlet_prior = dirichlet_prior

#         self.std_prior = nn.Parameter(torch.rand(out_features))
#         self.expert_weights_prior = nn.Parameter(torch.rand(monteclora_n))
#         self.gaussian_var_prior = torch.eye(out_features).to(device)
#         self.expert_weights = torch.ones(monteclora_n, device=device) / monteclora_n

#         self.sampler = None
#         if self.mc_training:
#             self.sampler = AsyncMonteCLoRASampler(self, buffer_size, device)
#             self.sampler.start()

#     def wishart_reparameterization(self, std, z_wishart):

#         updated_var = std @ z_wishart @ std.T
#         updated_var = torch.diag(torch.clip(updated_var.diag(), min=eps))
#         return updated_var

#     def multivariate_reparameterization(self, z_mvn, cov_matrix):

#         # L = torch.linalg.cholesky(cov_matrix).to(cov_matrix.dtype)
#         with torch.amp.autocast('cuda', enabled=False):
#             L = torch.linalg.cholesky(cov_matrix.float())
#         L = L.to(cov_matrix.dtype)
#         varsum = z_mvn @ L#torch.einsum('eio,op->eip', z_mvn, L) 
#         varsum = torch.nan_to_num(varsum, nan=eps)
#         return self.sample_scaler * varsum

#     def dirichlet_reparameterization(self, alpha, z_dirichlet):
#         eps = 1e-6
#         alpha = torch.clamp(alpha, min=eps)
        
#         mu = torch.log(alpha) - torch.log(alpha).mean()
#         diag_term = 1/alpha * (1 - 2/self.monteclora_n) + 1/(self.monteclora_n**2) * (1/alpha).sum()
#         diag_term = torch.clamp(diag_term, min=eps)
#         sigma = torch.diag(diag_term)
#         sigma = sigma + eps * torch.eye(len(alpha), device=self.device)
        
#         with torch.amp.autocast('cuda', enabled=False):
#             try:
#                 L = torch.linalg.cholesky(sigma.float())
#             except torch.linalg.LinAlgError:
#                 sigma = sigma + 1e-4 * torch.eye(len(alpha), device=self.device)
#                 L = torch.linalg.cholesky(sigma.float())
        
#         L = L.to(sigma.dtype)
#         return L @ z_dirichlet + mu

#     def calculate_entropy(self, expert_weights):
#         return (expert_weights ** 2).sum()

#     def dirichlet_kl(self, alpha2):
#         alpha1 = torch.tensor([self.dirichlet_prior]*self.monteclora_n, device=self.device)
#         gamma = lambda v: torch.lgamma(v).exp()
#         return torch.log(gamma(alpha2.sum())/gamma(alpha1.sum())) + \
#                (torch.log(gamma(alpha2)/gamma(alpha1))).sum() + \
#                ((alpha2 - alpha1)*(torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()

#     def wishart_kl(self, std):
#         var = std @ std.T
#         var = torch.diag(var.diag())
#         return 0.5 * (-torch.log(var).trace()*self.out_features + var.trace()*self.out_features - self.out_features**2)

#     def multivariate_kl(self, var):
#         var = torch.clamp(var, min=1e-6)
#         return self.monteclora_n * 0.5 * (var.trace() - torch.log(var).trace() - self.out_features)

#     def get_variational_loss(self):
#         if self.mc_training and self.training:
#             kl1 = self.dirichlet_kl(torch.exp(self.expert_weights_prior))
#             kl2 = self.wishart_kl(torch.diag(torch.exp(self.std_prior)))
#             kl3 = self.multivariate_kl(self.gaussian_var_prior)
#             entropy = self.calculate_entropy(self.expert_weights) if self.use_entropy else 0
#             # print(self.kl_loss_weight * (kl1 + kl2 + kl3), entropy)
#             return self.kl_loss_weight * (kl1 + kl2 + kl3), entropy
#         return 0, 0

#     def forward(self):
#         if self.training and self.mc_training:
#             # t = time.time()
#             sample = self.sampler.get() if self.sampler else None

#             if sample is not None:
#                 z_mvn = sample['z_mvn']
#                 z_wishart = sample['z_wishart']
#                 z_dirichlet = sample['z_dirichlet']
#             else:
#                 z_mvn = torch.randn((self.monteclora_n, self.in_features, self.out_features), device=self.device)
#                 wishart_sampler = Wishart(df=self.out_features, scale_tril=torch.eye(self.out_features, device=self.device))
#                 z_wishart = wishart_sampler._bartlett_sampling(torch.Size())
#                 z_dirichlet = torch.randn(self.monteclora_n, device=self.device)
#             # temp = time.time() - t
#             # print(temp)

#             # t = time.time()
#             std = torch.diag(torch.exp(self.std_prior))
#             gaussian_var = self.wishart_reparameterization(std, z_wishart)
#             self.gaussian_var_prior = gaussian_var

#             var = self.multivariate_reparameterization(z_mvn, gaussian_var)

#             expert_weights = self.dirichlet_reparameterization(torch.exp(self.expert_weights_prior), z_dirichlet)
#             expert_weights = torch.sigmoid(expert_weights)
#             expert_weights = expert_weights / expert_weights.sum()
#             self.expert_weights = expert_weights
#             # temp2 = time.time()  - t
#             # print(temp2)
#             # print(temp2/temp)
#             print(var.shape)
#             print(self.expert_weights.shape)
#             weighted_var = torch.einsum('n,nio->io', expert_weights, var) 
#             print(weighted_var.shape)
#             return var, self.expert_weights
#         else:
#             return -1, -1

#     def eval(self):
#         if self.sampler:
#             self.sampler.stop()
#             self.sampler = None
#         super().eval()

#     def __del__(self):
#         if hasattr(self, 'sampler') and self.sampler:
#             self.sampler.stop()


# Non Async Sampler
# class MonteCLoRASampler(nn.Module):
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         num_experts: int,
#         fan_in_fan_out : bool = False, 
#         use_entropy = True,
#         dirichlet_prior = 1,
#         sample_scaler = 3e-4,
#         kl_loss_weight = 1e-5,
#         mc_training = True,
#         device = 'cuda' if torch.cuda.is_available() else 'cpu',
#         **kwargs
#     ):
#         super(MonteCLoRASampler, self).__init__()
#         self.num_experts = num_experts
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fan_in_fan_out = fan_in_fan_out
#         self.use_entropy = use_entropy
#         self.device = device
#         self.kl_loss_weight = kl_loss_weight

#         self.mc_training = mc_training

#         self.wishart_df = out_features
#         self.wishert_prior = torch.eye(out_features)
#         self.dirichlet_prior = dirichlet_prior
#         self.sample_scaler = sample_scaler
#         self.expert_weights_prior = nn.Parameter(torch.rand(num_experts))
#         self.expert_weights = (torch.ones(num_experts)/ num_experts) 
#         self.std_prior = nn.Parameter(torch.rand(out_features))
#         self.gaussian_var_prior  = torch.eye(out_features)
#         self.sample_counter = 0
#         self.forward_counter = 0

#     def gamma(self, v):
#         return torch.lgamma(v).exp()
        
#     def multivariate_reparameterization(self, var2):
#         # https://www.wikiwand.com/en/Multivariate_normal_distribution#Drawing_values_from_the_distribution
#         sampler = MultivariateNormal(loc=torch.zeros(self.out_features).to(self.device), \
#                                      covariance_matrix=torch.eye(self.out_features).to(self.device))
#         all_vars = sampler.sample((self.num_experts, self.in_features)).to(self.device)
#         L = torch.linalg.cholesky(var2).to(var2.dtype)
#         varsum = torch.einsum('eio,op->eip', all_vars, L)
#         varsum = torch.nan_to_num(varsum, nan=eps)
#         return self.sample_scaler * varsum
        
#     def multivariate_kl(self, var):
#         # https://statproofbook.github.io/P/mvn-kl.html
#         # log-sum inequality - https://mat.hjg.com.ar/tic/img/lecture3.pdf
#         var = torch.clamp(var, min=1e-6)
#         return self.num_experts * 0.5 * (var.trace() - torch.log(var).trace() - self.out_features)
        
#     def wishart_reparameterization(self, std):

#         # http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/mvahtmlnode40.html
#         sampler = Wishart(df=self.wishart_df, scale_tril=torch.eye(self.out_features).to(self.device))

#         sample = sampler._bartlett_sampling(torch.Size()).to(torch.float32)

#         updated_var =  std @ sample.to(self.device) @ std.T
#         updated_var = torch.diag(torch.clip(updated_var.diag(),min=eps)).to(updated_var.device).to(torch.float32)
#         updated_var = torch.nan_to_num(updated_var, nan=eps)
#         return updated_var
        
#     def wishart_kl(self, std):
#         var = (std @ std.T)
#         var = torch.diag(var.diag()).to(var.device)
#         return 0.5 * (-torch.log(var).trace()*self.wishart_df + var.trace()*self.wishart_df - self.wishart_df**2)

#     def dirichlet_reparameterization(self, alpha2):
#         # https://arxiv.org/pdf/1703.01488
#         sampler = MultivariateNormal(loc=torch.zeros(self.num_experts).to(self.device), \
#                                      covariance_matrix=torch.eye(self.num_experts).to(self.device))
#         sample = sampler.sample().to(self.device)
#         mu = torch.log(alpha2) - 1/self.num_experts * torch.log(alpha2).sum()
#         sigma = torch.diag(1/alpha2 * (1 - 2/self.num_experts) + 1/(self.num_experts ** 2) * (1/alpha2).sum())
#         return torch.linalg.cholesky(sigma) @ sample + mu
        
#     def dirichlet_kl(self, alpha2):
#         # https://statproofbook.github.io/P/dir-kl.html
#         alpha1 = torch.tensor([self.dirichlet_prior]*self.num_experts).to(self.device)
#         kld = torch.log(self.gamma(alpha2.sum())/self.gamma(alpha1.sum())) + (torch.log(self.gamma(alpha2)/self.gamma(alpha1))).sum() + \
#               ((alpha2 - alpha1)*(torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()
#         return kld

#     def get_variational_loss(self):
#         if self.mc_training == True and self.training == True:
#             # kld of product of independent variables - http://www.math.tau.ac.il/~mansour/advanced-agt+ml/scribe5-lower-bound-MAB.pdf
#             kl1 = self.dirichlet_kl(torch.exp(self.expert_weights_prior))
#             kl2 = self.wishart_kl(torch.diag(torch.exp(self.std_prior)))
#             #kl2 = self.wishart_kl(self.std_prior)
#             kl3 = self.multivariate_kl(self.gaussian_var_prior)
            
#             if self.use_entropy == True:
#                 loss4 = self.calculate_entropy(self.expert_weights)
#                 return self.kl_loss_weight*(kl1 + kl2 + kl3), loss4
#                 #return loss4
#             else:
#                 return self.kl_loss_weight * (kl1 + kl2 + kl3), torch.tensor(0)
#                 #return 0
#         else:
#             return 0, 0
        
#     def calculate_entropy(self, expert_weights):
#         #return (expert_weights * expert_weights.log()).sum()
#         return (expert_weights ** 2).sum()

#     def forward(self):       
#         if self.training == True:            
#             if self.mc_training == True: 

#                 gaussian_var_prior = self.wishart_reparameterization(torch.diag(torch.exp(self.std_prior))) #*self.sample_scaler
#                 gaussian_var_prior = torch.nan_to_num(gaussian_var_prior, nan=eps)
#                 self.gaussian_var_prior = gaussian_var_prior

#                 var = self.multivariate_reparameterization(gaussian_var_prior)
#                 expert_weights = self.dirichlet_reparameterization(torch.exp(self.expert_weights_prior))
#                 expert_weights = nn.Sigmoid()(expert_weights)
#                 expert_weights = expert_weights/expert_weights.sum()
#                 self.expert_weights = expert_weights
#                 return var, expert_weights
#         else:
#             return -1,-1
        
#     def eval(self):
#         self.training = False

def _find_minimal_target_modules(
    target_modules: list[str] | set[str], other_module_names: list[str] | set[str]
) -> set[str]:
    """Find the minimal set of target modules that is sufficient to separate them from the other modules.

    Sometimes, a very large list of target_modules could be passed, which can slow down loading of adapters (e.g. when
    loaded from diffusers). It may be possible to condense this list from hundreds of items to just a handful of
    suffixes that are sufficient to distinguish the target modules from the other modules.

    Example:
        ```py
        >>> from peft.tuners.tuners_utils import _find_minimal_target_modules

        >>> target_modules = [f"model.decoder.layers.{i}.self_attn.q_proj" for i in range(100)]
        >>> target_modules += [f"model.decoder.layers.{i}.self_attn.v_proj" for i in range(100)]
        >>> other_module_names = [f"model.encoder.layers.{i}.self_attn.k_proj" for i in range(100)]
        >>> _find_minimal_target_modules(target_modules, other_module_names)
        {"q_proj", "v_proj"}
        ```

    Args:
        target_modules (`list[str]` | `set[str]`):
            The list of target modules.
        other_module_names (`list[str]` | `set[str]`):
            The list of other module names. They must not overlap with the target modules.

    Returns:
        `set[str]`:
            The minimal set of target modules that is sufficient to separate them from the other modules.

    Raises:
        ValueError:
            If `target_modules` is not a list or set of strings or if it contains an empty string. Also raises an error
            if `target_modules` and `other_module_names` contain common elements.
    """
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError("target_modules should be a list or set of strings.")

    target_modules = set(target_modules)
    if "" in target_modules:
        raise ValueError("target_modules should not contain an empty string.")

    other_module_names = set(other_module_names)
    if not target_modules.isdisjoint(other_module_names):
        msg = (
            "target_modules and other_module_names contain common elements, this should not happen, please "
            "open a GitHub issue at https://github.com/huggingface/peft/issues with the code to reproduce this issue"
        )
        raise ValueError(msg)

    # it is assumed that module name parts are separated by a "."
    def generate_suffixes(s):
        parts = s.split(".")
        return [".".join(parts[i:]) for i in range(len(parts))][::-1]

    # Create a reverse lookup for other_module_names to quickly check suffix matches
    other_module_suffixes = {suffix for item in other_module_names for suffix in generate_suffixes(item)}

    # Find all potential suffixes from target_modules
    target_modules_suffix_map = {item: generate_suffixes(item) for item in target_modules}

    # Initialize a set for required suffixes
    required_suffixes = set()

    # We sort the target_modules_suffix_map simply to get deterministic behavior, since sets have no order. In theory
    # the order should not matter but in case there is a bug, it's better for the bug to be deterministic.
    for item, suffixes in sorted(target_modules_suffix_map.items(), key=lambda tup: tup[1]):
        # Go through target_modules items, shortest suffixes first
        for suffix in suffixes:
            # If the suffix is already in required_suffixes or matches other_module_names, skip it
            if suffix in required_suffixes or suffix in other_module_suffixes:
                continue
            # Check if adding this suffix covers the item
            if not any(item.endswith("." + req_suffix) for req_suffix in required_suffixes):
                required_suffixes.add(suffix)
                break

    if not required_suffixes:
        return set(target_modules)
    return required_suffixes


class _ExcludedModule:
    """
    A private helper method used to represent excluded modules in the check_target_module_exists function.
    """

    def __bool__(self):
        return False


def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return _ExcludedModule()
        elif key in config.exclude_modules:
            return _ExcludedModule()
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            return _ExcludedModule()

    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    elif key in config.target_modules:
        # this module is specified directly in target_modules
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
            # For now, empty layers_pattern means any layer pattern is ok
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

    return target_module_found


def inspect_matched_modules(tuner: BaseTuner, adapter_name: str = "default") -> dict:
    """
    A helper function to inspect the set of matched and unmatched modules for a PEFT model and the given adapter.
    """
    config = tuner.peft_config[adapter_name]
    key_list = [key for key, _ in tuner.model.named_modules()]
    module_dict = {"matched": [], "unmatched": []}
    for key in key_list:
        if tuner._check_target_module_exists(config, key):
            module_dict["matched"].append(key)
        else:
            module_dict["unmatched"].append(key)
    return module_dict


def _maybe_include_all_linear_layers(peft_config: PeftConfig, model: nn.Module) -> PeftConfig:
    """
    Helper function to update `target_modules` to all linear/Conv1D layers if provided as 'all-linear'. Adapted from
    the QLoRA repository: https://github.com/artidoro/qlora/blob/main/qlora.py
    """
    if not hasattr(peft_config, "target_modules"):
        return peft_config

    # if `target_modules` is a string, convert to lower case and check if it matches "all-linear"
    if not (
        isinstance(peft_config.target_modules, str)
        and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND
    ):
        return peft_config

    if not isinstance(model, PreTrainedModel):
        raise ValueError(
            f"Only instances of PreTrainedModel support `target_modules={INCLUDE_LINEAR_LAYERS_SHORTHAND!r}`"
        )

    linear_classes = (torch.nn.Linear, Conv1D)

    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            names = name.rsplit(".", 1)[-1]  # get the base name
            linear_module_names.add(names)

    # Try to remove linear layers that should not be targeted as best as possible. We have to rely on convention as
    # there are no hard rules to detect these modules.
    module_names_to_exclude = set()
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        # ignore the last classification head for text generation models
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
        module_names_to_exclude.add(last_module_name)
    elif peft_config.task_type == TaskType.SEQ_CLS:
        # ignore classifier head for classification models (issue 2027)
        # there is no fix name for the classifier head, so check the common ones
        for name in SEQ_CLS_HEAD_NAMES:
            cls_head = getattr(model, name, None)
            if cls_head is not None:
                last_module_name = [name for name, module in model.named_modules() if module is cls_head][0]
                module_names_to_exclude.add(last_module_name)
                break

    linear_module_names -= module_names_to_exclude
    peft_config.target_modules = linear_module_names
    return peft_config


def check_adapters_to_merge(module: BaseTunerLayer, adapter_names: Optional[list[str]] = None) -> list[str]:
    """
    Helper function to check which adapters should be merged.

    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.

    """
    if adapter_names is None:
        adapter_names = module.active_adapters
    if isinstance(adapter_names, str):
        raise ValueError(f"adapter_names should be a list of strings, got {adapter_names!r}.")

    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        if adapter_names:
            warnings.warn(
                f"Already following adapters were merged {','.join(module.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names


def clone_module(module: nn.Module, share_weights=False):
    """Clone a module in a pytorch model.

    Clones a module of a model, optionally sharing all the parameters between the original and the clone. Simplifies
    reusing a module when manipulating the architecture of a model.
    """
    clone = copy.deepcopy(module)

    def _share_weights(src: nn.Module, dst: nn.Module):
        for name, param in src.named_parameters(recurse=False):
            dst.register_parameter(name, param)

    if share_weights:
        for name, submodule in module.named_modules():
            _share_weights(submodule, clone.get_submodule(name))

    return clone


def replicate_layers(model: nn.Module, layer_map: list[tuple[int, int]]):
    """Replicate layers in a transfomer model with weight sharing.

    This function looks for a module list attribute at model[(.model)*].layers and replicates the layers in the module
    list according to the layer map. For example the map `[[0, 4], [2, 5]]` will take the set of layers `[0, 1, 2, 3,
    4]` and replace them with a module list containing `[0, 1, 2, 3, 2, 3, 4]`.
    """
    while hasattr(model, "model"):
        model = model.model
    # Some variants of the bert model nest the main model under the bert attribute.
    if hasattr(model, "bert"):
        model = model.bert

    model_type = None
    layers: nn.ModuleList = None
    if hasattr(model, "layers"):
        model_type = "llama"
        layers = model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        model_type = "bert"
        layers = model.encoder.layer
    elif hasattr(model, "h"):
        model_type = "falcon"
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "Could not locate the layers attribute in the model. "
            "Expected Llama, Bert or Falcon compatible architectures."
        )

    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_module(layers[i], share_weights=True))
            # This is a hack needed to work around the layer_idx introduced in HF transformers.
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, "layer_idx"):
                    submodule.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == "llama":
        model.layers = layers
    elif model_type == "bert":
        model.encoder.layer = layers
    elif model_type == "falcon":
        model.h = layers
    else:
        raise ValueError("Unexpected model type, need to handle post-processing of layers.")
    if hasattr(model.config, "num_hidden_layers"):  # Common to Llama, Bert, Falcon.
        model.config.num_hidden_layers = len(new_layers)


# class AsyncMonteCLoRASampler(threading.Thread):
#     def __init__(self, model, buffer_size=10, device='cpu'):
#         super().__init__(daemon=True)
#         self.model = model
#         self.device = device
#         self.buffer_size = 10
#         self.queue = queue.Queue(maxsize=buffer_size)
#         self.running = True

#     def run(self):
#         while self.running:
#             if self.queue.qsize() < self.buffer_size:
#                 # print(self.queue.qsize())
#                 try:
#                     with torch.no_grad():
#                         z_mvn = torch.randn(
#                             (self.model.monteclora_n, self.model.in_features, self.model.out_features),
#                             device=self.device
#                         )
#                         wishart_sampler = Wishart(
#                             df=self.model.out_features,
#                             scale_tril=torch.eye(self.model.out_features, device=self.device)
#                         )
#                         z_wishart = wishart_sampler._bartlett_sampling(torch.Size())

#                         z_dirichlet = torch.randn(self.model.monteclora_n, device=self.device)

#                         sample = {
#                             'z_mvn': z_mvn,
#                             'z_wishart': z_wishart,
#                             'z_dirichlet': z_dirichlet
#                         }

#                         self.queue.put_nowait(sample)
#                 except KeyboardInterrupt:
#                     sys.exit(0)
#                 except Exception as e:
#                     print(f"[AsyncMonteCLoRASampler] Error: {e}")
#             else:
#                 time.sleep(0.01)

#     def get(self):
#         try:
#             return self.queue.get_nowait()
#         except queue.Empty:
#             return None

#     def stop(self):
#         self.running = False
#         self.join(timeout=1.0)

class AsyncMonteCLoRASampler(threading.Thread):
    def __init__(self, model, buffer_size=10, device='cpu'):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.buffer_size = 10
        self.queue = queue.Queue(maxsize=self.buffer_size)
        self.running = True
        self.wish_sampler = Wishart(
                            df=self.model.out_features,
                            scale_tril=torch.eye(self.model.out_features, device=self.device)
                        )

    def run(self):
        while self.running:
            if self.queue.qsize() < self.buffer_size:
                # print(self.queue.qsize())
                try:
                    with torch.no_grad():
                        z_mvn = torch.randn(
                            (self.model.monteclora_n, self.model.in_features, self.model.out_features),
                            device=self.device
                        )
                        
                        z_wishart = self.wish_sampler._bartlett_sampling(torch.Size())

                        z_dirichlet = torch.randn(self.model.monteclora_n, device=self.device)

                        sample = {
                            'z_mvn': z_mvn,
                            'z_wishart': z_wishart,
                            'z_dirichlet': z_dirichlet
                        }

                        self.queue.put_nowait(sample)
                except KeyboardInterrupt:
                    sys.exit(0)
                except Exception as e:
                    print(f"[AsyncMonteCLoRASampler] Error: {e}")
            else:
                time.sleep(0.01)

    def get(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.join(timeout=1.0)


class ApproximateMonteCLoRASampler:
    def __init__(self, model, buffer_size=500, device='cpu'):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        
        # Pre-generate ALL samples at initialization
        with torch.no_grad():
            self.z_mvn_buffer = torch.randn(
                (buffer_size, model.monteclora_n, model.in_features, model.out_features),
                device=device
            )
            self.z_dirichlet_buffer = torch.randn(
                (buffer_size, model.monteclora_n),
                device=device
            )
            
            # Pre-generate Wishart samples (this is the expensive part)
            wish_sampler = Wishart(
                df=model.out_features,
                scale_tril=torch.eye(model.out_features, device=device)
            )
            
            self.z_wishart_buffer = []
            for _ in range(buffer_size):
                self.z_wishart_buffer.append(
                    wish_sampler._bartlett_sampling(torch.Size())
                )
        
        self.index = 0

    def get(self):
        if self.index >= self.buffer_size:
            self.index = 0  # Cycle through buffer
        
        sample = {
            'z_mvn': self.z_mvn_buffer[self.index],
            'z_wishart': self.z_wishart_buffer[self.index],
            'z_dirichlet': self.z_dirichlet_buffer[self.index]
        }
        
        self.index += 1
        return sample

    def stop(self):
        pass

class BufferedMonteCLoRASampler:
    def __init__(self, model, buffer_size=200, device='cpu'):
        self.model = model
        self.device = device
        self.buffer_size = 15
        self.buffer = []
        self.index = 0
        
        # Pre-create Wishart sampler
        self.wish_sampler = Wishart(
            df=self.model.out_features,
            scale_tril=torch.eye(self.model.out_features, device=self.device)
        )
        
        # Pre-fill buffer
        self._refill_buffer()

    def _refill_buffer(self):
        """Generate samples in bulk"""
        with torch.no_grad():
            # Generate all z_mvn samples at once
            z_mvn_bulk = torch.randn(
                (self.buffer_size, self.model.monteclora_n, self.model.in_features, self.model.out_features),
                device=self.device
            )
            
            # Generate all z_dirichlet samples at once
            z_dirichlet_bulk = torch.randn(
                (self.buffer_size, self.model.monteclora_n),
                device=self.device
            )
            
            # Create buffer
            self.buffer = []
            for i in range(self.buffer_size):
                z_wishart = self.wish_sampler._bartlett_sampling(torch.Size())
                
                sample = {
                    'z_mvn': z_mvn_bulk[i],
                    'z_wishart': z_wishart,
                    'z_dirichlet': z_dirichlet_bulk[i]
                }
                self.buffer.append(sample)
        
        self.index = 0

    def get(self):
        if self.index >= self.buffer_size:
            self._refill_buffer()
        
        sample = self.buffer[self.index]
        self.index += 1
        return sample

    def stop(self):
        pass  # No cleanup needed

class MonteCLoRASampler(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        monteclora_n,
        fan_in_fan_out=False,
        use_entropy=True,
        dirichlet_prior=1,
        sample_scaler=3e-4,
        kl_loss_weight=1e-5,
        mc_training=True,
        buffer_size=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        monteclora_m = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.monteclora_n = monteclora_n
        self.fan_in_fan_out = fan_in_fan_out
        self.use_entropy = use_entropy
        self.device = device
        self.sample_scaler = sample_scaler
        self.kl_loss_weight = kl_loss_weight
        self.mc_training = mc_training
        self.dirichlet_prior = dirichlet_prior

        self.std_prior = nn.Parameter(torch.rand(out_features))
        self.expert_weights_prior = nn.Parameter(torch.rand(monteclora_n))
        self.gaussian_var_prior = torch.eye(out_features).to(device)
        self.expert_weights = torch.ones(monteclora_n, device=device) / monteclora_n

        self.sampler = None
        if self.mc_training:
            self.sampler = BufferedMonteCLoRASampler(self, 150, device)

    def wishart_reparameterization(self, std, z_wishart):

        updated_var = std @ z_wishart @ std.T
        updated_var = torch.diag(torch.clip(updated_var.diag(), min=eps))
        return updated_var

    def multivariate_reparameterization(self, z_mvn, cov_matrix):

        # L = torch.linalg.cholesky(cov_matrix).to(cov_matrix.dtype)
        with torch.amp.autocast('cuda', enabled=False):
            L = torch.linalg.cholesky(cov_matrix.float())
        L = L.to(cov_matrix.dtype)
        varsum = torch.einsum('eio,op->eip', z_mvn, L) #z_mvn @ L#
        varsum = torch.nan_to_num(varsum, nan=eps)
        return self.sample_scaler * varsum

    def dirichlet_reparameterization(self, alpha, z_dirichlet):
        eps = 1e-6
        alpha = torch.clamp(alpha, min=eps)
        
        mu = torch.log(alpha) - torch.log(alpha).mean()
        diag_term = 1/alpha * (1 - 2/self.monteclora_n) + 1/(self.monteclora_n**2) * (1/alpha).sum()
        diag_term = torch.clamp(diag_term, min=eps)
        sigma = torch.diag(diag_term)
        sigma = sigma + eps * torch.eye(len(alpha), device=self.device)
        
        with torch.amp.autocast('cuda', enabled=False):
            try:
                L = torch.linalg.cholesky(sigma.float())
            except torch.linalg.LinAlgError:
                sigma = sigma + 1e-4 * torch.eye(len(alpha), device=self.device)
                L = torch.linalg.cholesky(sigma.float())
        
        L = L.to(sigma.dtype)
        return L @ z_dirichlet + mu

    def calculate_entropy(self, expert_weights):
        return (expert_weights ** 2).sum()

    def dirichlet_kl(self, alpha2):
        alpha1 = torch.tensor([self.dirichlet_prior]*self.monteclora_n, device=self.device)
        gamma = lambda v: torch.lgamma(v).exp()
        return torch.log(gamma(alpha2.sum())/gamma(alpha1.sum())) + \
               (torch.log(gamma(alpha2)/gamma(alpha1))).sum() + \
               ((alpha2 - alpha1)*(torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()

    def wishart_kl(self, std):
        var = std @ std.T
        var = torch.diag(var.diag())
        return 0.5 * (-torch.log(var).trace()*self.out_features + var.trace()*self.out_features - self.out_features**2)

    def multivariate_kl(self, var):
        var = torch.clamp(var, min=1e-6)
        return self.monteclora_n * 0.5 * (var.trace() - torch.log(var).trace() - self.out_features)

    def get_variational_loss(self):
        if self.mc_training and self.training:
            kl1 = self.dirichlet_kl(torch.exp(self.expert_weights_prior))
            kl2 = self.wishart_kl(torch.diag(torch.exp(self.std_prior)))
            kl3 = self.multivariate_kl(self.gaussian_var_prior)
            entropy = self.calculate_entropy(self.expert_weights) if self.use_entropy else 0
            # print(self.kl_loss_weight * (kl1 + kl2 + kl3), entropy)
            return self.kl_loss_weight * (kl1 + kl2 + kl3), entropy
        return 0, 0

    def forward(self):
        if self.training and self.mc_training:
            t = time.time()
            sample = self.sampler.get() if self.sampler else None

            if sample is not None:
                z_mvn = sample['z_mvn']
                z_wishart = sample['z_wishart']
                z_dirichlet = sample['z_dirichlet']
            else:
                z_mvn = torch.randn((self.monteclora_n, self.in_features, self.out_features), device=self.device)
                z_wishart = self.sampler.wish_sampler._bartlett_sampling(torch.Size())
                z_dirichlet = torch.randn(self.monteclora_n, device=self.device)
            temp = time.time() - t
            # print(temp)

            # t = time.time()
            std = torch.diag(torch.exp(self.std_prior))
            gaussian_var = self.wishart_reparameterization(std, z_wishart)
            self.gaussian_var_prior = gaussian_var

            var = self.multivariate_reparameterization(z_mvn, gaussian_var)

            expert_weights = self.dirichlet_reparameterization(torch.exp(self.expert_weights_prior), z_dirichlet)
            expert_weights = torch.softmax(expert_weights, dim = -1)
            # expert_weights = expert_weights / expert_weights.sum()
            self.expert_weights = expert_weights
            return var, self.expert_weights
        else:
            return -1, -1

    def eval(self):
        if self.sampler:
            self.sampler.stop()
            self.sampler = None
        super().eval()

    def __del__(self):
        if hasattr(self, 'sampler') and self.sampler:
            self.sampler.stop()

class MonteCLoRALinear(nn.Linear):
    def __init__(
        self, 
        in_features, 
        out_features, 
        monteclora_n,
        fan_in_fan_out=False,
        use_entropy=True,
        dirichlet_prior=1,
        sample_scaler=3e-4,
        kl_loss_weight=1e-5,
        mc_training=True,
        buffer_size=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        monteclora_m = None
    ):
        nn.Linear.__init__(self, in_features, out_features, bias = False)
        self.in_features = in_features
        self.out_features = out_features
        self.monteclora_n = monteclora_n
        self.fan_in_fan_out = fan_in_fan_out
        self.use_entropy = use_entropy
        self.device = device
        self.sample_scaler = sample_scaler
        self.kl_loss_weight = kl_loss_weight
        self.mc_training = mc_training
        self.dirichlet_prior = dirichlet_prior

        self.std_prior = nn.Parameter(torch.rand(out_features))
        self.expert_weights_prior = nn.Parameter(torch.rand(monteclora_n))
        self.gaussian_var_prior = torch.eye(out_features).to(device)
        self.expert_weights = torch.ones(monteclora_n, device=device) / monteclora_n

        self.sampler = None
        if self.mc_training:
            self.sampler = BufferedMonteCLoRASampler(self, 150, device)
            # self.sampler.start()

        self.alpha1 = torch.tensor([self.dirichlet_prior]*self.monteclora_n, device=self.device)

    def wishart_reparameterization(self, std, z_wishart):

        updated_var = std @ z_wishart @ std.T
        updated_var = torch.diag(torch.clip(updated_var.diag(), min=eps))
        return updated_var

    def multivariate_reparameterization(self, z_mvn, cov_matrix):

        # L = torch.linalg.cholesky(cov_matrix).to(cov_matrix.dtype)
        with torch.amp.autocast('cuda', enabled=False):
            L = torch.linalg.cholesky(cov_matrix.float())
        L = L.to(cov_matrix.dtype)
        varsum = z_mvn @ L#torch.einsum('eio,op->eip', z_mvn, L) 
        varsum = torch.nan_to_num(varsum, nan=eps)
        return self.sample_scaler * varsum

    def dirichlet_reparameterization(self, alpha, z_dirichlet):
        alpha = torch.clamp(alpha, min=eps)
        
        mu = torch.log(alpha) - torch.log(alpha).mean()
        diag_term = 1/alpha * (1 - 2/self.monteclora_n) + 1/(self.monteclora_n**2) * (1/alpha).sum()
        diag_term = torch.clamp(diag_term, min=eps)
        sigma = torch.diag(diag_term)
        sigma = sigma + eps * torch.eye(len(alpha), device=self.device)
        
        with torch.amp.autocast('cuda', enabled=False):
            try:
                L = torch.linalg.cholesky(sigma.float())
            except torch.linalg.LinAlgError:
                sigma = sigma + 1e-4 * torch.eye(len(alpha), device=self.device)
                L = torch.linalg.cholesky(sigma.float())
        
        L = L.to(sigma.dtype)
        return L @ z_dirichlet + mu

    def calculate_entropy(self, expert_weights):
        return (expert_weights ** 2).sum()

    def dirichlet_kl(self, alpha2):
        gamma = lambda v: torch.lgamma(v).exp()
        return torch.log(gamma(alpha2.sum())/gamma(self.alpha1.sum())) + \
               (torch.log(gamma(alpha2)/gamma(self.alpha1))).sum() + \
               ((alpha2 - self.alpha1)*(torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()

    def wishart_kl(self, std):
        var = std @ std.T
        var = torch.diag(var.diag())
        return 0.5 * (-torch.log(var).trace()*self.out_features + var.trace()*self.out_features - self.out_features**2)

    def multivariate_kl(self, var):
        var = torch.clamp(var, min=1e-6)
        return self.monteclora_n * 0.5 * (var.trace() - torch.log(var).trace() - self.out_features)

    def get_variational_loss(self):
        if self.mc_training and self.training:
            kl1 = self.dirichlet_kl(torch.exp(self.expert_weights_prior))
            kl2 = self.wishart_kl(torch.diag(torch.exp(self.std_prior)))
            kl3 = self.multivariate_kl(self.gaussian_var_prior)
            entropy = self.calculate_entropy(self.expert_weights) if self.use_entropy else 0
            # print(self.kl_loss_weight * (kl1 + kl2 + kl3), entropy)
            return self.kl_loss_weight * (kl1 + kl2 + kl3), entropy
        return 0, 0

    def forward(self, x):
        if self.training and self.mc_training:
            # Get samples
            sample = self.sampler.get() if self.sampler else None

            if sample is not None:
                z_mvn = sample['z_mvn']
                z_wishart = sample['z_wishart']
                z_dirichlet = sample['z_dirichlet']
            else:
                z_mvn = torch.randn((self.monteclora_n, self.in_features, self.out_features), device=self.device)
                z_wishart = self.sampler.wish_sampler._bartlett_sampling(torch.Size())
                z_dirichlet = torch.randn(self.monteclora_n, device=self.device)

            # Reparameterization
            std = torch.diag(torch.exp(self.std_prior))
            gaussian_var = self.wishart_reparameterization(std, z_wishart)
            self.gaussian_var_prior = gaussian_var

            var = self.multivariate_reparameterization(z_mvn, gaussian_var)  # Shape: [monteclora_n, in_features, out_features]

            expert_weights = self.dirichlet_reparameterization(torch.exp(self.expert_weights_prior), z_dirichlet)
            expert_weights = torch.sigmoid(expert_weights)
            expert_weights = expert_weights / expert_weights.sum()
            self.expert_weights = expert_weights
            weighted_var = torch.einsum('n,nio->io', expert_weights, var)
            updated_weight = self.weight + self.sample_scaler * weighted_var.T
            if self.fan_in_fan_out:
                updated_weight = updated_weight.T
            output = F.linear(x, updated_weight, self.bias)
            
            return output
        else:
            return F.linear(x, self.weight, self.bias)

    def eval(self):
        if self.sampler:
            self.sampler.stop()
            self.sampler = None
        super().eval()

    def __del__(self):
        if hasattr(self, 'sampler') and self.sampler:
            self.sampler.stop()

