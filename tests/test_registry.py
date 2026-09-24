import pytest
from core.registry import Registry


def test_registry_registration_and_get():
    reg = Registry("test_reg")

    @reg.register()
    class DummyBlock:
        def __init__(self, val=1):
            self.val = val

    assert "DummyBlock" in reg
    cls = reg.get("DummyBlock")
    assert cls is DummyBlock

    inst = reg.build({"type": "DummyBlock", "val": 42})
    assert inst.val == 42


def test_registry_custom_name():
    reg = Registry("test_custom")

    @reg.register("custom_name")
    class Dummy:
        pass

    assert "custom_name" in reg
    assert reg.get("custom_name") is Dummy


def test_registry_duplicate_registration_error():
    reg = Registry("test_dup")

    @reg.register()
    class Duplicate:
        pass

    with pytest.raises(KeyError):
        @reg.register()
        class Duplicate:
            pass


def test_registry_build_missing_type():
    reg = Registry("test_missing")
    with pytest.raises(KeyError):
        reg.build({"no_type": 1})
