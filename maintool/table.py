#!/usr/bin/env python3
"""
extract_nn_structure_export.py
(已按用户要求修改：y 保存为整数，x 与参数使用 17 位浮点格式导出；
 输出格式：x/y/params 去掉 shape 行，负号替换为 ~，条目/样本用 1` 前缀，样本之间以 ++ 连接)
"""
import os
import sys
import json
import argparse
import traceback
import importlib.util
import importlib.machinery
from types import ModuleType


def import_module_from_path(path: str) -> ModuleType:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    module_name = os.path.splitext(os.path.basename(path))[0]
    pkgdir = os.path.dirname(path)

    # temporarily insert package dir to sys.path for relative imports in target module
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


def find_or_build_model(module):
    import torch.nn as nn

    # 1) look for a variable named 'model' which is an nn.Module instance
    if hasattr(module, "model") and isinstance(getattr(module, "model"), nn.Module):
        return getattr(module, "model")

    # 2) look for functions that likely build/return a model
    for fname in ("build_model", "get_model", "create_model", "make_model"):
        if hasattr(module, fname) and callable(getattr(module, fname)):
            try:
                m = getattr(module, fname)()
                if isinstance(m, nn.Module):
                    return m
            except Exception:
                pass

    # 3) find first nn.Module subclass defined in module and try to instantiate without args
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

# ---------------- 新增：构造 fresh (未训练) 模型的功能 ----------------
def build_fresh_model(module, trained_model=None, init_seed=None):
    """
    Try to create a freshly-initialized nn.Module based on the provided module.
    Strategy:
      1) call factory functions in module (build_model/get_model/...)
      2) instantiate nn.Module subclasses defined in module (no-arg)
      3) try to instantiate class(trained_model.__class__) with no args
      4) as a fallback, deepcopy trained_model and call reset_parameters() on submodules
    If init_seed is provided (int), set torch.manual_seed(init_seed) before calling factories/constructors.
    """
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
                # ignore and continue
                pass

    # 2) instantiate nn.Module subclasses defined in module
    classes = []
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                # prefer classes defined in this module (module name matches)
                if getattr(attr, "__module__", None) == getattr(module, "__name__", None):
                    classes.insert(0, attr)
                else:
                    classes.append(attr)
        except Exception:
            pass

    for cls in classes:
        try:
            # attempt no-arg constructor
            sig = None
            try:
                sig = inspect.signature(cls)
            except Exception:
                pass
            if sig:
                # if all parameters have defaults or there are zero params, try to call
                params = sig.parameters
                can_no_arg = all(
                    p.default is not inspect._empty or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    for p in params.values()
                )
                if not can_no_arg:
                    # try direct call anyway and catch exception
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
                        # ignore per-layer failures
                        pass
            return fresh
        except Exception:
            pass

    raise RuntimeError("Unable to construct a fresh (untrained) model from the provided module/model.")


def extract_layers(model, include_weights=False):
    import torch
    import torch.nn as nn

    layers = []
    for name, mod in model.named_modules():
        # skip root
        if name == "":
            continue
        info = {"name": name, "type": type(mod).__name__}
        try:
            if isinstance(mod, nn.Linear):
                info.update({
                    "in_features": mod.in_features,
                    "out_features": mod.out_features,
                    "weight_shape": list(mod.weight.shape),
                    "bias_shape": list(mod.bias.shape) if mod.bias is not None else None,
                })
                if include_weights:
                    info["weight"] = mod.weight.detach().cpu().numpy().tolist()
                    if mod.bias is not None:
                        info["bias"] = mod.bias.detach().cpu().numpy().tolist()
            elif isinstance(mod, nn.Conv2d):
                info.update({
                    "in_channels": mod.in_channels,
                    "out_channels": mod.out_channels,
                    "kernel_size": mod.kernel_size,
                    "stride": mod.stride,
                    "padding": mod.padding,
                    "weight_shape": list(mod.weight.shape),
                    "bias_shape": list(mod.bias.shape) if mod.bias is not None else None,
                })
                if include_weights:
                    info["weight"] = mod.weight.detach().cpu().numpy().tolist()
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                info.update({
                    "num_features": mod.num_features,
                    "weight_shape": list(mod.weight.shape) if getattr(mod, "weight", None) is not None else None,
                    "bias_shape": list(mod.bias.shape) if getattr(mod, "bias", None) is not None else None,
                })
            else:
                # try to collect common attributes if present
                for attr in ("kernel_size", "stride", "padding", "dilation", "groups"):
                    if hasattr(mod, attr):
                        info[attr] = getattr(mod, attr)
        except Exception:
            # If introspection fails, still record the basic info
            pass
        layers.append(info)
    return layers


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
    from torch.fx import symbolic_trace

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
    """Try to produce backward graph using torch.func.vjp and functorch.make_fx (if available)."""
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


# ------------------ 自定义序列化与保存函数 ------------------
def _format_scalar_for_output(val, float_fmt="%.17g"):
    """
    Format a scalar:
      - integer -> "123"
      - float  -> "0.1234" or "1.23e-05" using float_fmt
      - negative numbers: prefix with '~' instead of '-'
    """
    import numpy as _np

    # numpy scalar -> python scalar
    if isinstance(val, (_np.generic,)):
        try:
            val = val.item()
        except Exception:
            pass

    # integer types
    if isinstance(val, (int,)) or (hasattr(val, "dtype") and getattr(val, "dtype", None) in (_np.int32, _np.int64) if 'numpy' in str(type(val)) else False):
        return str(int(val))

    # try float
    try:
        f = float(val)
    except Exception:
        # fallback to repr
        s = repr(val)
        # replace leading minus if exists in a naive repr
        if s.startswith("-"):
            return "~" + s[1:]
        return s

    neg = (f < 0) or (str(f).startswith("-0"))
    formatted = float_fmt % (abs(f))
    # Ensure no leading plus sign, no spaces
    if neg:
        return "~" + formatted
    return formatted


def _serialize_array_like(obj, float_fmt="%.17g"):
    """
    Recursively serialize numpy array / list / nested lists into nested bracket string,
    with numbers formatted via _format_scalar_for_output.
    Example:
      [[-0.1, 0.2],[0.3, -0.4]] -> "[[~0.1,0.2],[0.3,~0.4]]"
    """
    import numpy as _np

    arr = _np.array(obj, copy=False)
    # recursion using ndarray
    def rec(a):
        if a.ndim == 0:
            return _format_scalar_for_output(a.item(), float_fmt=float_fmt)
        if a.ndim == 1:
            return "[" + ",".join(_format_scalar_for_output(x, float_fmt=float_fmt) for x in a.tolist()) + "]"
        # ndim >=2
        return "[" + ",".join(rec(a[i]) for i in range(a.shape[0])) + "]"

    return rec(arr)


def save_param_custom(path, tensor_like, float_fmt="%.17g"):
    """
    Save a parameter tensor to single-line like:
      1`[[~0.14,0.22],[...]]
    No shape header. Returns True on success.
    """
    try:
        s = _serialize_array_like(tensor_like, float_fmt=float_fmt)
        with open(path, "w", encoding="utf-8") as f:
            f.write(s + "\n")
        return True
    except Exception:
        return False


def save_dataset_custom(path, tensor_like, float_fmt="%.17g"):
    """
    Save a dataset (samples) using the format:
      1`[x0,x1,...]++1`[x0,x1,...]++...
    Each sample prefixed by 1`. If scalar label, still write as [val].
    No shape header. Return True on success.
    """
    import numpy as _np
    try:
        arr = _np.array(tensor_like)
        entries = []
        if arr.ndim == 0:
            # single scalar -> single entry
            entries.append("[" + _format_scalar_for_output(arr.item(), float_fmt=float_fmt) + "]")
        elif arr.ndim == 1:
            # treat each element as a sample scalar -> write [val]
            for i in range(arr.shape[0]):
                entries.append("[" + _format_scalar_for_output(arr[i].item() if hasattr(arr[i], "item") else arr[i], float_fmt=float_fmt) + "]")
        else:
            # arr.ndim >=2: iterate first axis as samples
            for i in range(arr.shape[0]):
                row = arr[i]
                # ensure row is 1D
                if getattr(row, "ndim", 1) > 1:
                    # nested row: serialize as nested list
                    s = _serialize_array_like(row, float_fmt=float_fmt)
                    entries.append("" + s)
                else:
                    # 1D row -> [a,b,c]
                    entries.append("[" + ",".join(_format_scalar_for_output(x, float_fmt=float_fmt) for x in row.tolist()) + "]")
        content = "++".join(entries)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        return True
    except Exception:
        return False
# ------------------ 新增结束 ------------------


def save_array_as_txt(path, tensor_like, float_fmt="%.17g"):
    """保留原实现（不改动），但不再作为 x/y/params 的主要输出函数（保持备份）"""
    import numpy as _np

    try:
        arr = tensor_like.detach().cpu().numpy() if hasattr(tensor_like, "detach") else _np.array(tensor_like)
        shape = list(arr.shape)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# shape: " + " ".join(str(x) for x in shape) + "\n")
            if arr.size == 0:
                return True

            # integer type?
            if _np.issubdtype(arr.dtype, _np.integer):
                int_fmt = "%d"
                if arr.ndim == 1:
                    for v in arr:
                        f.write(int_fmt % int(v) + "\n")
                    return True
                if arr.ndim == 2:
                    _np.savetxt(f, arr, fmt=int_fmt)
                    return True
                # >2D: flatten in rows
                flat = arr.reshape(-1)
                row_len = 16
                for i in range(0, flat.size, row_len):
                    row = flat[i:i + row_len]
                    f.write(" ".join(int_fmt % int(x) for x in row) + "\n")
                return True

            # float-like
            if arr.ndim == 1:
                for v in arr:
                    try:
                        f.write((float_fmt % float(v)) + "\n")
                    except Exception:
                        f.write(repr(v) + "\n")
                return True
            if arr.ndim == 2:
                _np.savetxt(f, arr, fmt=float_fmt)
                return True
            # >2D: flatten with rows of row_len items
            flat = arr.reshape(-1)
            row_len = 16
            for i in range(0, flat.size, row_len):
                row = flat[i:i + row_len]
                line = " ".join((float_fmt % float(x)) for x in row)
                f.write(line + "\n")
        return True
    except Exception:
        # on any failure, return False (caller may handle)
        return False


def infer_output_dim_by_forward(model, dummy_input):
    """Infer output class dimension by a forward pass on dummy_input.
    Returns an integer C (>=1).
    """
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

    # Fallback: try to find last Linear/Conv2d
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract NN layers + FX execution order to JSON and export tensors/weights (txt outputs)"
    )
    parser.add_argument("source", help="Path to python file defining the model")
    parser.add_argument("--out", help="Also write structure JSON to this path (optional)")
    parser.add_argument("--out-dir", help="Output folder for exported assets (default: <source>_export/)")
    parser.add_argument("--val-samples", type=int, default=100, help="Total samples to generate (train+test)")
    parser.add_argument("--val-seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--include-weights", action="store_true", help="Include weights in JSON (can be large)")
    parser.add_argument("--include-backward", action="store_true", help="Try to extract backward graph")
    parser.add_argument(
        "--label-mode", choices=["onehot", "int"], default="onehot",
        help="'onehot' for 0/1 matrix (C>1) or column (C==1); 'int' for integer class labels"
    )
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction reserved for test (0..1)")
    # 新增参数
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

    layers = extract_layers(model, include_weights=args.include_weights)
    mlp_info = extract_mlp_info(layers)
    dummy_input = build_dummy_input_from_model(model)
    execution = extract_fx_execution_order(model, dummy_input)

    backward_info = None
    if args.include_backward:
        try:
            backward_info = extract_backward_graph_if_possible(model, dummy_input)
        except Exception:
            backward_info = {
                "error": "exception when extracting backward",
                "traceback": traceback.format_exc(),
            }

    result = {
        "source": os.path.basename(src),
        "layers": layers,
        "mlp_summary": mlp_info,
        "execution": execution,
    }
    if backward_info is not None:
        result["backward"] = backward_info

    # Save structure JSON inside out_dir
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

    # Generate synthetic dataset
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

        # ---- 修改：produce integer y (int64). If onehot requested, produce integer one-hot (0/1) matrix.
        if args.label_mode == "onehot":
            if out_dim == 1:
                # column of 0/1 ints
                y_all = torch.randint(0, 2, (total_n, 1), dtype=torch.int64)
            else:
                labels = torch.randint(0, out_dim, (total_n,), dtype=torch.int64)
                y_all = torch.nn.functional.one_hot(labels, num_classes=out_dim).to(torch.int64)
        else:  # int labels
            if out_dim == 1:
                y_all = torch.randint(0, 2, (total_n,), dtype=torch.int64)
            else:
                y_all = torch.randint(0, out_dim, (total_n,), dtype=torch.int64)

        # Shuffle and split
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

        # Save to custom-format txt files (注意：x 与参数使用 17 位有效数字格式导出)
        x_train_path = os.path.join(out_dir, "x_train.txt")
        x_test_path = os.path.join(out_dir, "x_test.txt")
        y_train_path = os.path.join(out_dir, "y_train.txt")
        y_test_path = os.path.join(out_dir, "y_test.txt")

        # 使用自定义保存函数：去除 shape 第一行，负号替换为 ~，样本之间用 ++ 连接，样本前有 1`
        save_dataset_custom(x_train_path, x_train, float_fmt="%.17g")
        save_dataset_custom(x_test_path, x_test, float_fmt="%.17g")
        # y: 若为整数则写整数；若为向量/浮点则按浮点格式写出
        save_dataset_custom(y_train_path, y_train, float_fmt="%.17g")
        save_dataset_custom(y_test_path, y_test, float_fmt="%.17g")

        print("Saved train/test splits:")
        print("  ", x_train_path)
        print("  ", x_test_path)
        print("  ", y_train_path)
        print("  ", y_test_path)

    except Exception as e:
        print("Failed to generate/save synthetic train/test data:", e)

    # Export each parameter as a separate .txt file using custom format
    exported_files = []
    params_map_path = None
    try:
        for name, param in model.named_parameters():
            safe_name = name.replace(".", "__")
            filename = f"{safe_name}.txt"
            path = os.path.join(out_dir, filename)
            # ---- 修改：使用 17 位有效数字格式导出参数，且使用自定义单行格式（去除 shape，负号改 ~）
            if save_param_custom(path, param.detach().cpu().numpy(), float_fmt="%.17g"):
                try:
                    import numpy as _np
                    shape = list(_np.array(param.detach().cpu().numpy()).shape)
                except Exception:
                    shape = None
                exported_files.append({
                    "param_name": name,
                    "file": filename,
                    "shape": shape,
                    "initial": False
                })

        # 如果用户要求，导出 fresh (未训练) 参数
        if args.export_init:
            try:
                fresh_model = build_fresh_model(module, trained_model=model, init_seed=args.init_seed)
                # iterate fresh params and save with suffix __init
                for name, param in fresh_model.named_parameters():
                    safe_name = name.replace(".", "__")
                    filename = f"{safe_name}__init.txt"
                    path = os.path.join(out_dir, filename)
                    # param may be torch tensor
                    arr = param.detach().cpu().numpy() if hasattr(param, "detach") else param
                    if save_param_custom(path, arr, float_fmt="%.17g"):
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

    # Also save an index file describing exported assets (as txt)
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

    # Optionally also write to args.out if specified
    try:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("Wrote structure to:", args.out)
    except Exception as e:
        print("Failed to write output:", e)


if __name__ == "__main__":
    main()
