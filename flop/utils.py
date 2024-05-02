from typing import List, Union, Tuple
from copy import deepcopy

import torch.nn as nn

from flop.hardconcrete import HardConcrete
from flop.linear import (
    ProjectedLinear,
    HardConcreteProjectedLinear,
    HardConcreteLinear,
    ProjectedLinearWithMask,
)


def make_projected_linear(module: nn.Module, in_place: bool = True) -> nn.Module:
    """Replace all nn.Linear with ProjectedLinear.

    Parameters
    ----------
    module : nn.Module
        The input module to modify
    in_place : bool, optional
        Whether to modify in place, by default True

    Returns
    -------
    nn.Module
        The updated module.

    """
    # First find all nn.Linear modules
    modules = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            modules.append((name, child))
        else:
            make_projected_linear(child, in_place)

    # Replace all modules found
    new_module = module if in_place else deepcopy(module)
    for name, child in modules:
        new_child = ProjectedLinear.from_module(child)
        setattr(new_module, name, new_child)

    return new_module


def make_hard_concrete(
    module: nn.Module,
    in_place: bool = True,
    init_mean: float = 0.5,
    init_std: float = 0.01,
) -> nn.Module:
    """Replace all ProjectedLinear with HardConcreteProjectedLinear.

    Parameters
    ----------
    module : nn.Module
        The input module to modify
    in_place : bool, optional
        Whether to modify in place, by default True

    Returns
    -------
    nn.Module
        The updated module.

    """
    # First find all ProjectedLinear modules
    modules: List[Tuple[str, Union[ProjectedLinear, nn.Linear]]] = []
    for name, child in module.named_children():
        if isinstance(child, ProjectedLinear):
            modules.append((name, child))
        elif isinstance(child, nn.Linear):
            modules.append((name, child))
        else:
            make_hard_concrete(child, in_place, init_mean, init_std)

    # Replace all modules found
    new_module = module if in_place else deepcopy(module)
    for name, child in modules:
        if isinstance(child, ProjectedLinear):
            new_child = HardConcreteProjectedLinear.from_module(
                child, init_mean, init_std
            )
        else:  # must be nn.Linear
            new_child = HardConcreteLinear.from_module(child, init_mean, init_std)
        setattr(new_module, name, new_child)

    return new_module


def make_projected_linear_with_mask(
    module: nn.Module, in_place: bool = True, init_zero: bool = False
) -> nn.Module:
    """Replace all ProjectedLinear with ProjectedLinearWithMask.

    Parameters
    ----------
    module : nn.Module
        The input module to modify
    in_place : bool, optional
        Whether to modify in place, by default True

    Returns
    -------
    nn.Module
        The updated module.

    """
    # First find all ProjectedLinear modules
    modules = []
    for name, child in module.named_children():
        if isinstance(child, ProjectedLinear):
            modules.append((name, child))
        else:
            make_projected_linear_with_mask(child, in_place, init_zero=init_zero)

    # Replace all modules found
    new_module = module if in_place else deepcopy(module)
    for name, child in modules:
        new_child = ProjectedLinearWithMask.from_module(child, init_zero=init_zero)
        setattr(new_module, name, new_child)

    return new_module


def get_hardconcrete_linear_modules(
    module: nn.Module,
) -> List[Union[HardConcreteProjectedLinear, HardConcreteLinear]]:
    """Get all HardConcrete*Linear modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[nn.Module]
        A list of the HardConcrete*Linear module.

    """
    modules: List[Union[HardConcreteProjectedLinear, HardConcreteLinear]] = []
    for m in module.children():
        if isinstance(m, HardConcreteProjectedLinear):
            modules.append(m)
        elif isinstance(m, HardConcreteLinear):
            modules.append(m)
        else:
            modules.extend(get_hardconcrete_linear_modules(m))
    return modules


def get_hardconcrete_proj_linear_modules(
    module: nn.Module,
) -> List[HardConcreteProjectedLinear]:
    """Get all HardConcreteProjectedLinear modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[HardConcreteProjectedLinear]
        A list of the HardConcreteProjectedLinear module.

    """
    modules = []
    for m in module.children():
        if isinstance(m, HardConcreteProjectedLinear):
            modules.append(m)
        else:
            modules.extend(get_hardconcrete_proj_linear_modules(m))
    return modules


def get_hardconcrete_modules(module: nn.Module) -> List[HardConcrete]:
    """Get all HardConcrete modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[HardConcrete]
        A list of the HardConcrete module.

    """
    modules = []
    for m in module.children():
        if isinstance(m, HardConcrete):
            modules.append(m)
        else:
            modules.extend(get_hardconcrete_modules(m))
    return modules


def get_projected_linear_with_mask_modules(
    module: nn.Module,
) -> List[ProjectedLinearWithMask]:
    """Get all ProjectedLinearWithMask modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[HardConcreteProjectedLinear]
        A list of the ProjectedLinearWithMask module.

    """
    modules = []
    for m in module.children():
        if isinstance(m, ProjectedLinearWithMask):
            modules.append(m)
        else:
            modules.extend(get_projected_linear_with_mask_modules(m))
    return modules


def get_projected_linear_masks(module: nn.Module) -> List[nn.Parameter]:
    """Get all masks from ProjectedLinearWithMask modules.

    Parameters
    ----------
    module : nn.Module
        The input module

    Returns
    -------
    List[HardConcrete]
        A list of the masks.

    """
    modules = []
    for m in module.children():
        if isinstance(m, ProjectedLinearWithMask):
            modules.append(m.mask)
        else:
            modules.extend(get_projected_linear_masks(m))
    return modules


def get_num_prunable_params(modules) -> int:
    return sum([module.num_prunable_parameters() for module in modules])


def get_num_params(modules, train=True) -> int:
    return sum([module.num_parameters(train) for module in modules])
