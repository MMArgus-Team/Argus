"""Contract test for the "no sampling params" rule introduced by 17bd04845.

That commit stopped sending temperature/top_p/top_k on the wire and removed the
arguments from the *call sites*, but left ``temperature`` as a **required**
keyword-only parameter in the signatures of ``MemoryWriter._call_llm``,
``MemoryReviewer._call_llm``, ``RecallAgent._create_chat_completion`` and the
four ``MemoryLLMClient.call_chat`` implementations.  Every writer tick, every L2
/L3 aggregation, every reviewer wake and every recall decision therefore raised
``TypeError: _call_llm() missing 1 required keyword-only argument`` — and each of
those four paths swallows exceptions (``asyncio.gather(return_exceptions=True)``
for reviewer wave1, a bare ``except Exception`` for the writer loop and the
aggregators), so memory write / aggregate / review failed *silently*.

This test enforces both halves of the contract by AST-scanning the production
module, so a future half-fix in either direction fails in CI instead of in prod:

  1. none of those functions may *declare* a sampling parameter, and
  2. no call to them may *pass* one.
"""
import ast
import pathlib
import unittest

#: Functions that talk to an LLM and must not carry sampling params.
_GUARDED_FUNCS = {
    "call_chat",
    "_call_llm",
    "_create_chat_completion",
}

_SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}

_WORKERS = (
    pathlib.Path(__file__).resolve().parents[3]
    / "agent" / "multimodal" / "_workers.py"
)


def _load_tree() -> ast.Module:
    return ast.parse(_WORKERS.read_text(encoding="utf-8"), filename=str(_WORKERS))


def _all_arg_names(node) -> list:
    a = node.args
    out = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
    if a.vararg:
        out.append(a.vararg)
    if a.kwarg:
        out.append(a.kwarg)
    return [x.arg for x in out]


def _called_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


class TestNoSamplingParamsInSignatures(unittest.TestCase):
    def test_guarded_functions_declare_no_sampling_params(self):
        offenders = []
        for node in ast.walk(_load_tree()):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in _GUARDED_FUNCS:
                continue
            bad = _SAMPLING_PARAMS & set(_all_arg_names(node))
            if bad:
                offenders.append(
                    f"{_WORKERS.name}:{node.lineno} def {node.name}() "
                    f"declares {sorted(bad)}")
        self.assertEqual(
            offenders, [],
            "sampling params must not appear in these signatures — a required "
            "one turns every call site into a swallowed TypeError:\n  "
            + "\n  ".join(offenders))


class TestNoSamplingParamsAtCallSites(unittest.TestCase):
    def test_guarded_calls_pass_no_sampling_params(self):
        offenders = []
        for node in ast.walk(_load_tree()):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in _GUARDED_FUNCS:
                continue
            bad = {kw.arg for kw in node.keywords if kw.arg in _SAMPLING_PARAMS}
            if bad:
                offenders.append(
                    f"{_WORKERS.name}:{node.lineno} call {name}(...) "
                    f"passes {sorted(bad)}")
        self.assertEqual(
            offenders, [],
            "17bd04845 retired sampling params; do not reintroduce them:\n  "
            + "\n  ".join(offenders))


class TestCallSitesMatchSignatures(unittest.TestCase):
    """Catch the general shape of the 17bd04845 regression: a required
    keyword-only parameter that no call site supplies."""

    def test_no_required_kwonly_arg_is_missed_by_every_call_site(self):
        tree = _load_tree()
        defs = {}          # name -> (lineno, {required kwonly names})
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in _GUARDED_FUNCS:
                continue
            required = {
                a.arg for a, d in zip(node.args.kwonlyargs,
                                      node.args.kw_defaults)
                if d is None
            }
            # Several implementations share a name (4 call_chat overrides);
            # union of requirements is the strictest contract callers face.
            prev = defs.get(node.name)
            if prev is None:
                defs[node.name] = (node.lineno, set(required))
            else:
                prev[1].update(required)

        supplied = {name: set() for name in defs}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in supplied:
                continue
            if any(kw.arg is None for kw in node.keywords):
                continue  # **kwargs forwarding — cannot be checked statically
            supplied[name].update(
                kw.arg for kw in node.keywords if kw.arg is not None)

        offenders = []
        for name, (lineno, required) in defs.items():
            if not supplied[name]:
                continue  # no direct call sites in this module
            missing = required - supplied[name]
            if missing:
                offenders.append(
                    f"{_WORKERS.name}:{lineno} def {name}() requires "
                    f"{sorted(missing)} but no call site passes it")
        self.assertEqual(offenders, [], "\n  ".join(offenders))


if __name__ == "__main__":
    unittest.main()
