#!/usr/bin/env python3
"""
extract_nn_structure_export_cnn_tokens.py

CNN-aware exporter:
 - extracts model structure and optional forward/backward info
 - records per-module output shapes via forward hooks (best-effort)
 - exports parameters in custom format; for Conv2d weights/biases it writes multiple tokens
   (one token per output channel) joined with '++', each token prefixed by `1`
 - supports exporting fresh (untrained) parameters with --export-init
 - keeps dataset export, index and params_map generation

Numeric formatting: floats use '%.17g', negative sign replaced by '~'.
"""
import os
import sys
import json
import argparse
import traceback
import importlib.util
import importlib.machinery
from types import ModuleType


# ----------------- module import helpers -----------------
def import_module_from_path(path: str) -> ModuleType:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    module_name = os.path.splitext(os.path.basename(path))[0]
    pkgdir = os.path.dirname(path)

    added_to_syspath = False
    if pkgdir not in sys.path:
        sys.path.insert(0, pkgdir)
        added_to_syspath = True

    try:
        loader = importlib.machinery.SourceFileLoader(module_name, path)
        spec = importlib.util.spec_from_loader(module_name, loader)
        if spec is None:
            raise ImportError(f"Could not create spec for {path}")
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    except Exception as e:
        tb = traceback.format_exc()
        raise ImportError(f"Failed to import module from {path}:\n{tb}") from e
    finally:
        if added_to_syspath:
            try:
                sys.path.remove(pkgdir)
            except Exception:
                pass


# ----------------- find/build model -----------------
def find_or_build_model(module):
    import torch.nn as nn

    # 1) look for a variable named 'model' which is an nn.Module instance
    if hasattr(module, "model") and isinstance(getattr(module, "model"), nn.Module):
        return getattr(module, "model")

    # 2) look for factory functions
    for fname in ("build_model", "get_model", "create_model", "make_model"):
        if hasattr(module, fname) and callable(getattr(module, fname)):
            try:
                m = getattr(module, fname)()
                if isinstance(m, nn.Module):
                    return m
            except Exception:
                pass

    # 3) instantiate nn.Module subclasses defined in module (no-arg if possible)
    classes = []
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                classes.append(attr)
        except Exception:
            pass

    if classes:
        for cls in classes:
            try:
                inst = cls()
                if isinstance(inst, nn.Module):
                    return inst
            except Exception:
                continue

    # 4) check globals for any nn.Module instance
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            if isinstance(attr, nn.Module):
                return attr
        except Exception:
            pass

    raise RuntimeError(
        "Could not find or instantiate an nn.Module in the provided file. "
        "Please provide a module that defines a 'model' variable, "
        "or a 'build_model()/get_model()' factory, or a nn.Module subclass with no-arg constructor."
    )


# ----------------- build fresh model fallback -----------------
def build_fresh_model(module, trained_model=None, init_seed=None):
    import torch
    import torch.nn as nn
    import inspect
    import copy

    if init_seed is not None:
        try:
            torch.manual_seed(int(init_seed))
        except Exception:
            pass

    # 1) factory functions
    for fname in ("build_model", "get_model", "create_model", "make_model"):
        if hasattr(module, fname) and callable(getattr(module, fname)):
            try:
                m = getattr(module, fname)()
                if isinstance(m, nn.Module):
                    return m
            except Exception:
                pass

    # 2) instantiate nn.Module subclasses defined in module
    classes = []
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                if getattr(attr, "__module__", None) == getattr(module, "__name__", None):
                    classes.insert(0, attr)
                else:
                    classes.append(attr)
        except Exception:
            pass

    for cls in classes:
        try:
            sig = None
            try:
                sig = inspect.signature(cls)
            except Exception:
                pass
            if sig:
                params = sig.parameters
                can_no_arg = all(
                    p.default is not inspect._empty or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    for p in params.values()
                )
                if not can_no_arg:
                    inst = cls()
                else:
                    inst = cls()
            else:
                inst = cls()
            if isinstance(inst, nn.Module):
                return inst
        except Exception:
            continue

    # 3) try to instantiate class of trained_model if available
    if trained_model is not None:
        try:
            cls = trained_model.__class__
            try:
                inst = cls()
                if isinstance(inst, nn.Module):
                    return inst
            except Exception:
                pass
        except Exception:
            pass

    # 4) fallback: deepcopy trained_model and call reset_parameters() on submodules
    if trained_model is not None:
        try:
            fresh = copy.deepcopy(trained_model)
            for m in fresh.modules():
                if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
                    try:
                        m.reset_parameters()
                    except Exception:
                        pass
            return fresh
        except Exception:
            pass

    raise RuntimeError("Unable to construct a fresh (untrained) model from the provided module/model.")


# ----------------- layer extraction -----------------
def _to_serializable(x):
    if isinstance(x, tuple):
        return list(x)
    return x


def extract_layers(model, include_weights=False):
    import torch
    import torch.nn as nn

    layers = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        info = {"name": name, "type": type(mod).__name__}
        try:
            if isinstance(mod, nn.Linear):
                info.update({
                    "in_features": getattr(mod, "in_features", None),
                    "out_features": getattr(mod, "out_features", None),
                    "weight_shape": list(getattr(mod, "weight").shape) if getattr(mod, "weight", None) is not None else None,
                    "bias_shape": list(getattr(mod, "bias").shape) if getattr(mod, "bias", None) is not None else None,
                })
                if include_weights and getattr(mod, "weight", None) is not None:
                    info["weight"] = getattr(mod, "weight").detach().cpu().numpy().tolist()
                    if getattr(mod, "bias", None) is not None:
                        info["bias"] = getattr(mod, "bias").detach().cpu().numpy().tolist()
            elif isinstance(mod, nn.Conv2d):
                info.update({
                    "in_channels": _to_serializable(getattr(mod, "in_channels", None)),
                    "out_channels": _to_serializable(getattr(mod, "out_channels", None)),
                    "kernel_size": _to_serializable(getattr(mod, "kernel_size", None)),
                    "stride": _to_serializable(getattr(mod, "stride", None)),
                    "padding": _to_serializable(getattr(mod, "padding", None)),
                    "dilation": _to_serializable(getattr(mod, "dilation", None)),
                    "groups": _to_serializable(getattr(mod, "groups", None)),
                    "weight_shape": list(getattr(mod, "weight").shape) if getattr(mod, "weight", None) is not None else None,
                    "bias_shape": list(getattr(mod, "bias").shape) if getattr(mod, "bias", None) is not None else None,
                })
                if include_weights and getattr(mod, "weight", None) is not None:
                    info["weight"] = getattr(mod, "weight").detach().cpu().numpy().tolist()
                    if getattr(mod, "bias", None) is not None:
                        info["bias"] = getattr(mod, "bias").detach().cpu().numpy().tolist()
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                info.update({
                    "num_features": getattr(mod, "num_features", None),
                    "weight_shape": list(getattr(mod, "weight").shape) if getattr(mod, "weight", None) is not None else None,
                    "bias_shape": list(getattr(mod, "bias").shape) if getattr(mod, "bias", None) is not None else None,
                })
            else:
                for attr in ("kernel_size", "stride", "padding", "dilation", "groups"):
                    if hasattr(mod, attr):
                        info[attr] = _to_serializable(getattr(mod, attr))
        except Exception:
            pass
        layers.append(info)
    return layers


# ----------------- formatting and export helpers -----------------
def _format_scalar_for_output(val, float_fmt="%.17g"):
    import numpy as _np

    if isinstance(val, (_np.generic,)):
        try:
            val = val.item()
        except Exception:
            pass

    if isinstance(val, (int,)):
        return str(int(val))

    try:
        f = float(val)
    except Exception:
        s = repr(val)
        if s.startswith("-"):
            return "~" + s[1:]
        return s

    neg = (f < 0) or (str(f).startswith("-0"))
    formatted = float_fmt % (abs(f))
    if neg:
        return "~" + formatted
    return formatted


def _serialize_array_like(obj, float_fmt="%.17g"):
    import numpy as _np
    arr = _np.array(obj, copy=False)

    def rec(a):
        if a.ndim == 0:
            return _format_scalar_for_output(a.item(), float_fmt=float_fmt)
        if a.ndim == 1:
            return "[" + ",".join(_format_scalar_for_output(x, float_fmt=float_fmt) for x in a.tolist()) + "]"
        # ndim >= 2
        return "[" + ",".join(rec(a[i]) for i in range(a.shape[0])) + "]"

    return rec(arr)


def save_param_custom(path, tensor_like, float_fmt="%.17g"):
    """Save entire tensor as single-token file (1`<array>)"""
    try:
        s = "1`" + _serialize_array_like(tensor_like, float_fmt=float_fmt)
        with open(path, "w", encoding="utf-8") as f:
            f.write(s + "\n")
        return True
    except Exception:
        return False


def save_param_tokens(path, tensor_like, float_fmt="%.17g"):
    """
    Save tensor as multiple tokens separated by '++'. Iterate over axis 0.
    - For conv weight [out_ch, in_ch, kh, kw] produce out_ch tokens (each token is the kernel array)
      * If in_ch == 1 we squeeze the in-channel dimension so token is 2D (kh x kw)
      * If in_ch > 1 token will be a list of in_ch 2D kernels
    - For conv bias [out_ch] produce out_ch tokens (each token is [bias_scalar])
    - For general tensors, iterate axis 0 and serialize subarrays.
    """
    import numpy as _np
    try:
        a = _np.array(tensor_like, copy=False)
        if a.ndim == 0:
            tok = "1`[" + _format_scalar_for_output(a.item(), float_fmt=float_fmt) + "]"
            with open(path, "w", encoding="utf-8") as f:
                f.write(tok + "\n")
            return True

        tokens = []
        for i in range(a.shape[0]):
            sub = a[i]
            sub_arr = _np.array(sub, copy=False)
            # If sub_arr represents conv kernel with an extra in_channel dim ==1, squeeze it
            if sub_arr.ndim >= 3 and sub_arr.shape[0] == 1:
                try:
                    sub_arr = _np.squeeze(sub_arr, axis=0)
                except Exception:
                    pass
            if sub_arr.ndim == 0:
                tok = "1`[" + _format_scalar_for_output(sub_arr.item(), float_fmt=float_fmt) + "]"
            else:
                tok = "1`" + _serialize_array_like(sub_arr, float_fmt=float_fmt)
            tokens.append(tok)

        content = "++".join(tokens)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        return True
    except Exception:
        return False


def save_dataset_custom(path, tensor_like, float_fmt="%.17g"):
    import numpy as _np
    try:
        arr = _np.array(tensor_like)
        entries = []
        if arr.ndim == 0:
            entries.append("1`[" + _format_scalar_for_output(arr.item(), float_fmt=float_fmt) + "]")
        elif arr.ndim == 1:
            for i in range(arr.shape[0]):
                entries.append("1`[" + _format_scalar_for_output(arr[i].item() if hasattr(arr[i], "item") else arr[i], float_fmt=float_fmt) + "]")
        else:
            for i in range(arr.shape[0]):
                row = arr[i]
                if getattr(row, "ndim", 1) > 1:
                    s = _serialize_array_like(row, float_fmt=float_fmt)
                    entries.append("1`" + s)
                else:
                    entries.append("1`[" + ",".join(_format_scalar_for_output(x, float_fmt=float_fmt) for x in row.tolist()) + "]")
        content = "++".join(entries)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        return True
    except Exception:
        return False


def save_dataset_custom2(path, tensor_like, float_fmt="%.17g"):
    import numpy as _np
    try:
        arr = _np.array(tensor_like)

        # Flattening the input to 1x784 (flatten the image)
        arr = arr.reshape(arr.shape[0], -1)  # Flatten each example to 784 dimensions

        entries = []
        for i in range(arr.shape[0]):
            # Format each flattened input as 1`[[784]] (wrap each in [[...]] with a 1` prefix)
            entries.append(
                f"1`[[{','.join(map(lambda x: _format_scalar_for_output(x, float_fmt=float_fmt), arr[i]))}]]")

        # Join all entries with '++' separator
        content = "++".join(entries)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        return True
    except Exception:
        return False


# ----------------- dummy input and fx helpers -----------------
def build_dummy_input_from_model(model):
    import torch
    from torch import nn as _nn

    first = None
    for _, mod in model.named_modules():
        if isinstance(mod, (_nn.Linear, _nn.Conv2d)):
            first = mod
            break

    if first is None:
        return torch.randn(1, 4)
    if isinstance(first, _nn.Linear):
        return torch.randn(1, first.in_features)
    if isinstance(first, _nn.Conv2d):
        return torch.randn(1, first.in_channels, 32, 32)
    return torch.randn(1, 4)


def extract_fx_execution_order(model, dummy_input):
    try:
        from torch.fx import symbolic_trace
    except Exception:
        return {"error": "torch.fx not available"}

    try:
        gm = symbolic_trace(model)
    except Exception as e:
        return {"error": f"symbolic_trace failed: {repr(e)}"}

    nodes = []
    for node in gm.graph.nodes:
        def arg_to_str(a):
            try:
                if hasattr(a, "shape"):
                    return f"<Tensor shape={getattr(a, 'shape')}>"
                return str(a)
            except Exception:
                return repr(a)
        nodes.append({
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": [arg_to_str(a) for a in node.args],
            "kwargs": {k: arg_to_str(v) for k, v in node.kwargs.items()},
        })
    return {"graph_code": gm.code, "nodes": nodes}


def extract_backward_graph_if_possible(model, dummy_input):
    try:
        try:
            from torch.func import vjp
        except Exception:
            from functorch import vjp
        try:
            from functorch import make_fx
        except Exception:
            return {"error": "make_fx not available (functorch required to build backward graph)"}

        y, pullback = vjp(model, dummy_input)

        def backward_only(grad_output):
            grads = pullback(grad_output)
            if isinstance(grads, (tuple, list)):
                return grads[0]
            return grads

        import torch
        ones = torch.ones_like(y)
        bwd_gm = make_fx(backward_only)(ones)

        code = getattr(bwd_gm, "code", None)
        nodes = []
        for node in bwd_gm.graph.nodes:
            def arg_to_str(a):
                try:
                    if hasattr(a, "shape"):
                        return f"<Tensor shape={getattr(a, 'shape')}>"
                    return str(a)
                except Exception:
                    return repr(a)
            nodes.append({
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": [arg_to_str(a) for a in node.args],
                "kwargs": {k: arg_to_str(v) for k, v in node.kwargs.items()},
            })
        return {"code": code, "nodes": nodes}
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": f"failed to extract backward graph: {repr(e)}", "traceback": tb}


# ----------------- output-shape recording via hooks -----------------
def record_module_output_shapes(model, dummy_input, device=None):
    import torch
    shapes = {}
    hooks = []

    try:
        device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    except Exception:
        import torch as _torch
        device = _torch.device("cpu")

    def make_hook(name):
        def hook(module, inp, out):
            try:
                if hasattr(out, "shape"):
                    shapes[name] = tuple(int(x) for x in out.shape)
                else:
                    shapes[name] = None
            except Exception:
                shapes[name] = None
        return hook

    for name, mod in model.named_modules():
        if name == "":
            continue
        try:
            h = mod.register_forward_hook(make_hook(name))
            hooks.append(h)
        except Exception:
            pass

    try:
        model.eval()
        with torch.no_grad():
            dummy = dummy_input.to(device)
            model.to(device)
            _ = model(dummy)
    except Exception:
        pass
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
    return shapes


# ----------------- utility -----------------
def extract_mlp_info(layers):
    mlp_layers = []
    for l in layers:
        t = l.get("type", "").lower()
        if t in ("linear", "dense"):
            mlp_layers.append({
                "layer_name": l.get("name"),
                "in_features": l.get("in_features") or l.get("input_dim") or None,
                "out_features": l.get("out_features") or l.get("units") or None,
            })
    return {"total_mlp_layers": len(mlp_layers), "mlp_layers": mlp_layers}


def infer_output_dim_by_forward(model, dummy_input):
    import torch
    try:
        model.eval()
        with torch.no_grad():
            y = model(dummy_input)
        if isinstance(y, (tuple, list)) and len(y) > 0:
            y = y[0]
        if hasattr(y, "shape"):
            if y.dim() >= 2:
                return int(y.shape[-1])
            return 1
    except Exception:
        pass

    try:
        import torch.nn as nn
        last_linear = None
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is not None:
            return int(last_linear.out_features)
        last_conv = None
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                last_conv = mod
        if last_conv is not None:
            return int(last_conv.out_channels)
    except Exception:
        pass
    return 1


# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser(description="Extract NN layers + FX execution order to JSON and export tensors/weights (CNN-aware, conv tokens)")
    parser.add_argument("source", help="Path to python file defining the model")
    parser.add_argument("--out", help="Also write structure JSON to this path (optional)")
    parser.add_argument("--out-dir", help="Output folder for exported assets (default: <source>_export/)")
    parser.add_argument("--val-samples", type=int, default=100, help="Total samples to generate (train+test)")
    parser.add_argument("--val-seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--include-weights", action="store_true", help="Include weights in JSON (can be large)")
    parser.add_argument("--include-backward", action="store_true", help="Try to extract backward graph")
    parser.add_argument("--label-mode", choices=["onehot", "int"], default="onehot")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--export-init", action="store_true", help="Also export parameters of a freshly initialized (untrained) model")
    parser.add_argument("--init-seed", type=int, default=None, help="If set, seed RNG before initializing fresh model (for reproducibility)")
    args = parser.parse_args()

    src = os.path.abspath(args.source)
    if not os.path.isfile(src):
        print("Source file not found:", src)
        sys.exit(2)

    out_dir = args.out_dir or (os.path.splitext(src)[0] + "_export")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print("Failed to create output directory:", out_dir, e)
        sys.exit(7)

    try:
        module = import_module_from_path(src)
    except Exception as e:
        print("Failed to import module:", e)
        sys.exit(3)

    try:
        model = find_or_build_model(module)
    except Exception as e:
        print("Failed to find or build model:", e)
        sys.exit(4)

    import torch

    model.eval()

    # extract layer descriptions
    layers = extract_layers(model, include_weights=args.include_weights)
    mlp_info = extract_mlp_info(layers)
    dummy_input = build_dummy_input_from_model(model)
    execution = extract_fx_execution_order(model, dummy_input)

    # capture forward output shapes
    try:
        shapes = record_module_output_shapes(model, dummy_input)
    except Exception:
        shapes = {}

    for l in layers:
        nm = l.get("name")
        if nm in shapes:
            l["output_shape"] = list(shapes[nm]) if shapes[nm] is not None else None

    backward_info = None
    if args.include_backward:
        try:
            backward_info = extract_backward_graph_if_possible(model, dummy_input)
        except Exception:
            backward_info = {"error": "exception when extracting backward", "traceback": traceback.format_exc()}

    result = {
        "source": os.path.basename(src),
        "layers": layers,
        "mlp_summary": mlp_info,
        "execution": execution,
    }
    if backward_info is not None:
        result["backward"] = backward_info

    structure_json_path = os.path.join(out_dir, "structure.json")

    try:
        if not args.include_weights:
            for l in result.get("layers", []):
                if "weight" in l:
                    del l["weight"]
                if "bias" in l:
                    del l["bias"]
        with open(structure_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Wrote structure JSON to:", structure_json_path)
    except Exception as e:
        print("Failed to write structure JSON:", e)

    # generate synthetic dataset
    x_train_path = None
    x_test_path = None
    y_train_path = None
    y_test_path = None

    try:
        torch.manual_seed(args.val_seed)

        sample_shape = list(dummy_input.shape)[1:]
        total_n = max(1, int(args.val_samples))
        x_all = torch.randn(total_n, *sample_shape)

        out_dim = max(1, int(infer_output_dim_by_forward(model, dummy_input)))

        if args.label_mode == "onehot":
            if out_dim == 1:
                y_all = torch.randint(0, 2, (total_n, 1), dtype=torch.int64)
            else:
                labels = torch.randint(0, out_dim, (total_n,), dtype=torch.int64)
                y_all = torch.nn.functional.one_hot(labels, num_classes=out_dim).to(torch.int64)
        else:
            if out_dim == 1:
                y_all = torch.randint(0, 2, (total_n,), dtype=torch.int64)
            else:
                y_all = torch.randint(0, out_dim, (total_n,), dtype=torch.int64)

        idx = torch.randperm(total_n)
        test_n = int(total_n * float(args.test_split))
        if test_n <= 0:
            test_n = 1 if total_n > 1 else 0
        if test_n >= total_n:
            test_n = total_n - 1 if total_n > 1 else total_n
        test_idx = idx[:test_n]
        train_idx = idx[test_n:]

        x_test = x_all[test_idx]
        x_train = x_all[train_idx]
        y_test = y_all[test_idx]
        y_train = y_all[train_idx]



        x_train_path = os.path.join(out_dir, "x_train.txt")
        x_test_path = os.path.join(out_dir, "x_test.txt")
        y_train_path = os.path.join(out_dir, "y_train.txt")
        y_test_path = os.path.join(out_dir, "y_test.txt")

        save_dataset_custom2(x_train_path, x_train, float_fmt="%.17g")
        save_dataset_custom2(x_test_path, x_test, float_fmt="%.17g")
        save_dataset_custom(y_train_path, y_train, float_fmt="%.17g")
        save_dataset_custom(y_test_path, y_test, float_fmt="%.17g")

        print("Saved train/test splits:")
        print("  ", x_train_path)
        print("  ", x_test_path)
        print("  ", y_train_path)
        print("  ", y_test_path)

    except Exception as e:
        print("Failed to generate/save synthetic train/test data:", e)

    # export parameters
    exported_files = []
    params_map_path = None
    try:
        for name, param in model.named_parameters():
            safe_name = name.replace(".", "__")
            filename = f"{safe_name}.txt"
            path = os.path.join(out_dir, filename)
            arr = param.detach().cpu().numpy() if hasattr(param, "detach") else param

            lower = name.lower()
            try:
                # ---- Modified logic: export conv *weights* as tokens (per out channel),
                #                      but export conv *biases* as a single array (same as fc bias).
                if ("conv" in lower or "conv2d" in lower):
                    if "weight" in lower:
                        # conv weight -> keep multi-token per out_channel (existing behavior)
                        ok = save_param_tokens(path, arr, float_fmt="%.17g")
                    elif "bias" in lower:
                        # conv bias -> export as single token file (same format as fc bias)
                        ok = save_param_custom(path, arr, float_fmt="%.17g")
                    else:
                        # fallback for other conv params (unlikely)
                        ok = save_param_custom(path, arr, float_fmt="%.17g")
                else:
                    # non-conv params: preserve existing heuristic from your script
                    if (("weight" in lower) or ("bias" in lower)) and (
                            hasattr(arr, 'ndim') and arr.ndim >= 3 and arr.shape[0] > 1):
                        ok = save_param_tokens(path, arr, float_fmt="%.17g")
                    else:
                        ok = save_param_custom(path, arr, float_fmt="%.17g")
            except Exception:
                ok = save_param_custom(path, arr, float_fmt="%.17g")

        # export initial params if requested
        if args.export_init:
            try:
                fresh_model = build_fresh_model(module, trained_model=model, init_seed=args.init_seed)
                for name, param in fresh_model.named_parameters():
                    safe_name = name.replace(".", "__")
                    filename = f"{safe_name}__init.txt"
                    path = os.path.join(out_dir, filename)
                    arr = param.detach().cpu().numpy() if hasattr(param, "detach") else param

                    lower = name.lower()
                    try:
                        if (("weight" in lower) or ("bias" in lower)) and (("conv" in lower) or ("conv2d" in lower)):
                            ok = save_param_tokens(path, arr, float_fmt="%.17g")
                        else:
                            if hasattr(arr, 'ndim') and arr.ndim >= 3 and arr.shape[0] > 1:
                                ok = save_param_tokens(path, arr, float_fmt="%.17g")
                            else:
                                ok = save_param_custom(path, arr, float_fmt="%.17g")
                    except Exception:
                        ok = save_param_custom(path, arr, float_fmt="%.17g")

                    if ok:
                        try:
                            import numpy as _np
                            shape = list(_np.array(arr).shape)
                        except Exception:
                            shape = None
                        exported_files.append({
                            "param_name": name,
                            "file": filename,
                            "shape": shape,
                            "initial": True
                        })
                print("Exported initial (untrained) parameters (--export-init).")
            except Exception as e:
                print("Failed to build/export initial model parameters:", e)

        params_map_path = os.path.join(out_dir, "params_map.txt")
        with open(params_map_path, "w", encoding="utf-8") as f:
            json.dump(exported_files, f, indent=2, ensure_ascii=False)
        print("Wrote params_map to:", params_map_path)
    except Exception as e:
        print("Failed to export parameters:", e)

    # augment structure with params files for convenience
    try:
        param_to_file = {e["param_name"]: e["file"] for e in exported_files}
        for l in result.get("layers", []):
            layer_name = l.get("name", "")
            params_files = []
            for p_name, fname in param_to_file.items():
                if p_name.startswith(layer_name + ".") or p_name.startswith(layer_name + "__"):
                    params_files.append({"param_name": p_name, "file": fname})
            if not params_files:
                for p_name, fname in param_to_file.items():
                    if layer_name and (layer_name in p_name):
                        params_files.append({"param_name": p_name, "file": fname})
            if params_files:
                l["params_files"] = params_files
    except Exception:
        pass

    # rewrite final structure json
    try:
        with open(structure_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Updated structure JSON with parameter file links:", structure_json_path)
    except Exception as e:
        print("Failed to update structure JSON:", e)

    # write index
    index = {
        "structure_file": os.path.basename(structure_json_path),
        "dataset": {
            "x_train": os.path.basename(x_train_path) if x_train_path else None,
            "x_test": os.path.basename(x_test_path) if x_test_path else None,
            "y_train": os.path.basename(y_train_path) if y_train_path else None,
            "y_test": os.path.basename(y_test_path) if y_test_path else None,
        },
        "params_map": os.path.basename(params_map_path) if params_map_path else None,
        "exported_params": exported_files,
    }
    index_path = os.path.join(out_dir, "index.txt")
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print("Wrote export index to:", index_path)
    except Exception as e:
        print("Failed to write export index:", e)

    # optionally also write to args.out
    try:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("Wrote structure to:", args.out)
    except Exception as e:
        print("Failed to write output:", e)


if __name__ == "__main__":
    main()
