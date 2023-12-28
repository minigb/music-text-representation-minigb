from abc import ABCMeta


class NonOverridableMeta(ABCMeta):
    def __new__(cls, name, bases, namespace):
        for base in bases:
            if isinstance(base, NonOverridableMeta):
                for method in getattr(base, "_non_overridable_methods_", set()):
                    if method in namespace and method in base._non_overridable_methods_:
                        raise TypeError(f"Cannot override non-overridable method '{method}' in {name}")
        return super().__new__(cls, name, bases, namespace)