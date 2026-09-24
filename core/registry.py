from typing import Any, Callable, Dict, Optional, Union


class Registry:
    """
    A registry to map string keys to classes or functions.
    Supports decorator registration, lookup, and config-based instantiation.
    """

    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Any] = {}

    def register(
        self,
        name: Optional[Union[str, Callable]] = None,
        module_cls: Optional[Any] = None,
        force: bool = False,
    ) -> Union[Any, Callable]:
        """
        Register a module class or function.

        Can be used as a decorator:
            @REGISTRY.register()
            class MyModule: ...

            @REGISTRY.register("custom_name")
            class MyModule: ...

            @REGISTRY.register
            class MyModule: ...

        Or as a direct function call:
            REGISTRY.register("name", MyModule)
        """
        # Case: @REGISTRY.register (without parens)
        if callable(name) and module_cls is None:
            cls = name
            self._do_register(cls.__name__, cls, force=force)
            return cls

        # Case: REGISTRY.register("name", MyModule) or REGISTRY.register(module_cls=MyModule)
        if module_cls is not None:
            target_name = name or module_cls.__name__
            self._do_register(target_name, module_cls, force=force)
            return module_cls

        # Case: @REGISTRY.register() or @REGISTRY.register("custom_name")
        def _decorator(cls):
            target_name = name or cls.__name__
            self._do_register(target_name, cls, force=force)
            return cls

        return _decorator

    def _do_register(self, name: str, module_cls: Any, force: bool = False):
        if not force and name in self._module_dict:
            raise KeyError(
                f"'{name}' is already registered in registry '{self.name}' "
                f"by {self._module_dict[name]}."
            )
        self._module_dict[name] = module_cls

    def get(self, name: str) -> Any:
        if name not in self._module_dict:
            available = list(self._module_dict.keys())
            raise KeyError(
                f"'{name}' not found in registry '{self.name}'. Available: {available}"
            )
        return self._module_dict[name]

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, items={list(self._module_dict.keys())})"

    def build(self, cfg: Dict[str, Any], **default_args) -> Any:
        """
        Instantiate an object from a config dictionary with a 'type' key.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a dict, got {type(cfg)}")
        if "type" not in cfg:
            raise KeyError(f"Config for registry '{self.name}' must contain key 'type'")

        cfg = cfg.copy()
        obj_type = cfg.pop("type")
        obj_cls = self.get(obj_type)

        kwargs = default_args.copy()
        kwargs.update(cfg)

        return obj_cls(**kwargs)


# Global registry instances
MODULES = Registry("modules")
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
MODELS = Registry("models")
LOSSES = Registry("losses")
TRANSFORMS = Registry("transforms")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
