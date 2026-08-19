"""Regression guard for config type coercion (finding C9).

build_config used to ``setattr(cfg, key, mm[key])`` with the raw YAML value,
so a numeric hyperparameter written as a string (common in YAML) silently
polluted the Config with a str. The fix coerces to the field's annotated type
and skips (with a warning) on failure, keeping the dataclass default.
"""
from agent.multimodal.hermes_glue import _COERCE_FAILED, _coerce_config_value


def test_int_from_string():
    assert _coerce_config_value("k", "5", "int") == 5
    assert isinstance(_coerce_config_value("k", "5", "int"), int)


def test_float_from_string():
    assert _coerce_config_value("k", "1.5", "float") == 1.5
    assert isinstance(_coerce_config_value("k", "1.5", "float"), float)


def test_bool_from_string_truthy_and_falsy():
    assert _coerce_config_value("k", "true", "bool") is True
    assert _coerce_config_value("k", "on", "bool") is True
    assert _coerce_config_value("k", "1", "bool") is True
    assert _coerce_config_value("k", "false", "bool") is False
    assert _coerce_config_value("k", "no", "bool") is False
    assert _coerce_config_value("k", "", "bool") is False


def test_bool_not_confused_with_int():
    # bool is a subclass of int; ensure a real int type coerces to int not bool.
    assert _coerce_config_value("k", True, "int") == 1
    assert type(_coerce_config_value("k", True, "int")) is int


def test_invalid_int_reports_failed():
    assert _coerce_config_value("k", "abc", "int") is _COERCE_FAILED
    assert _coerce_config_value("k", "n/a", "float") is _COERCE_FAILED


def test_str_field_stringifies():
    assert _coerce_config_value("k", 5, "str") == "5"
    assert _coerce_config_value("k", "x", "str") == "x"


def test_unknown_annotation_passes_through():
    obj = object()
    assert _coerce_config_value("k", obj, "SomeComplexType") is obj


def test_build_config_coerces_string_numeric():
    """End-to-end: a stringified numeric mm value lands as the correct type."""
    from dataclasses import fields
    import agent.multimodal.hermes_glue as glue
    from agent.multimodal._config import Config

    # Pick a real int-annotated numeric key that build_config coerces.
    int_keys = [
        f.name for f in fields(Config)
        if (f.type == "int" or getattr(f.type, "__name__", "") == "int")
        and f.name in glue._NUMERIC_KEYS
    ]
    assert int_keys, "expected at least one int numeric key to test"
    key = int_keys[0]

    cfg = glue.build_config({"multimodal": {key: "7"}})
    assert getattr(cfg, key) == 7
    assert type(getattr(cfg, key)) is int
