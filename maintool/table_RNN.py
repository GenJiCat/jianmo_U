import os
import sys
import json
import argparse
import traceback
import importlib.util
import importlib.machinery
from types import ModuleType
import torch
import torch.nn as nn
import numpy as np

# --------- 格式化数值输出 ---------
def _format_scalar_for_output(val, float_fmt="%.5g"):
    """
    格式化标量值为字符串（浮动或整数），以便用于输出
    """
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


# --------- 模型加载和提取 ---------
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


def find_or_build_model(module):
    import torch.nn as nn
    if hasattr(module, "model") and isinstance(getattr(module, "model"), nn.Module):
        return getattr(module, "model")
    for fname in ("build_model", "get_model", "create_model", "make_model"):
        if hasattr(module, fname) and callable(getattr(module, fname)):
            try:
                m = getattr(module, fname)()
                if isinstance(m, nn.Module):
                    return m
            except Exception:
                pass
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


# --------- LSTM 层提取 ---------
def extract_lstm_layers(model, include_weights=False):
    import torch.nn as nn
    lstm_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LSTM):
            info = {
                "name": name,
                "type": type(mod).__name__,
                "input_size": mod.input_size,
                "hidden_size": mod.hidden_size,
                "num_layers": mod.num_layers,
                "batch_first": mod.batch_first,
            }
            if include_weights:
                # 提取权重和偏置
                weight_ih = mod.weight_ih_l0.detach().cpu().numpy()
                weight_hh = mod.weight_hh_l0.detach().cpu().numpy()
                bias_ih = mod.bias_ih_l0.detach().cpu().numpy() if mod.bias_ih_l0 is not None else None
                bias_hh = mod.bias_hh_l0.detach().cpu().numpy() if mod.bias_hh_l0 is not None else None

                # 将权重矩阵拆分为各个门
                input_size = mod.input_size
                hidden_size = mod.hidden_size

                # 处理权重矩阵，拆分为每个门的权重
                info["W_ii"] = weight_ih[:hidden_size, :]  # 输入门
                info["W_if"] = weight_ih[hidden_size:2 * hidden_size, :]  # 遗忘门
                info["W_io"] = weight_ih[2 * hidden_size:3 * hidden_size, :]  # 输出门
                info["W_ic"] = weight_ih[3 * hidden_size:, :]  # 候选细胞状态

                info["W_hi"] = weight_hh[:hidden_size, :]  # 隐状态到隐状态的权重（输入门）
                info["W_hf"] = weight_hh[hidden_size:2 * hidden_size, :]  # 隐状态到隐状态的权重（遗忘门）
                info["W_ho"] = weight_hh[2 * hidden_size:3 * hidden_size, :]  # 隐状态到隐状态的权重（输出门）
                info["W_hc"] = weight_hh[3 * hidden_size:, :]  # 隐状态到隐状态的权重（候选细胞状态）

                if bias_ih is not None:
                    info["bias_ii"] = bias_ih[:hidden_size]  # 输入门的偏置
                    info["bias_if"] = bias_ih[hidden_size:2 * hidden_size]  # 遗忘门的偏置
                    info["bias_io"] = bias_ih[2 * hidden_size:3 * hidden_size]  # 输出门的偏置
                    info["bias_ic"] = bias_ih[3 * hidden_size:]  # 候选细胞状态的偏置

                if bias_hh is not None:
                    info["bias_hi"] = bias_hh[:hidden_size]  # 输入门的偏置（隐状态到隐状态）
                    info["bias_hf"] = bias_hh[hidden_size:2 * hidden_size]  # 遗忘门的偏置（隐状态到隐状态）
                    info["bias_ho"] = bias_hh[2 * hidden_size:3 * hidden_size]  # 输出门的偏置（隐状态到隐状态）
                    info["bias_hc"] = bias_hh[3 * hidden_size:]  # 候选细胞状态的偏置（隐状态到隐状态）

            lstm_layers.append(info)
    return lstm_layers


# --------- 全连接层提取 ---------
def extract_linear_layers(model, include_weights=False):
    import torch.nn as nn
    linear_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            info = {
                "name": name,
                "type": type(mod).__name__,
                "in_features": mod.in_features,
                "out_features": mod.out_features,
            }
            if include_weights:
                # 提取权重和偏置
                weight = mod.weight.detach().cpu().numpy()
                bias = mod.bias.detach().cpu().numpy() if mod.bias is not None else None

                # 保存权重和偏置
                info["weight"] = weight
                if bias is not None:
                    info["bias"] = bias

            linear_layers.append(info)
    return linear_layers


# --------- 保存LSTM层权重 ---------
def save_lstm_weights_to_file(out_dir, lstm_layers):
    """
    保存每个LSTM层的权重到文件
    """
    for layer in lstm_layers:
        layer_name = layer['name']
        for key, value in layer.items():
            if 'W_' in key or 'bias' in key:  # 只保存与权重或偏置相关的键
                filename = f"{layer_name}_{key}.txt"
                path = os.path.join(out_dir, filename)
                if save_param_custom(path, value):
                    print(f"Successfully saved {key} of {layer_name} to {path}")
                else:
                    print(f"Failed to save {key} of {layer_name}.")
def _serialize_array_like(obj, float_fmt="%.5g"):
    """
    将numpy数组或列表递归序列化为字符串
    """
    arr = np.asarray(obj)

    def rec(a):
        if a.ndim == 0:
            return _format_scalar_for_output(a.item(), float_fmt=float_fmt)
        if a.ndim == 1:
            return "[" + ",".join(_format_scalar_for_output(x, float_fmt=float_fmt) for x in a.tolist()) + "]"
        return "[" + ",".join(rec(a[i]) for i in range(a.shape[0])) + "]"

    return rec(arr)

# --------- 保存权重和偏置 ---------
def save_param_custom(path, tensor_like, float_fmt="%.5g"):
    """
    保存权重或偏置参数到文件
    """
    try:
        s = _serialize_array_like(tensor_like, float_fmt=float_fmt)
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 确保目录存在
        with open(path, "w", encoding="utf-8") as f:
            f.write(s + "\n")
        print(f"Successfully saved: {path}")
        return True
    except Exception as e:
        print(f"Failed to save {path}: {str(e)}")
        return False


# --------- 保存标签数据 ---------
# --------- 保存标签数据 ---------
def save_y_labels(y_data, out_dir, filename="labels.txt"):
    """
    保存标签数据 y 为特定格式，增加唯一的索引
    """
    try:
        formatted_data = ""
        for idx, label in enumerate(y_data):
            # 添加标签的索引，格式为 [label, index]
            formatted_data += f"1[{_format_scalar_for_output(label)}," + str(idx) + "]++"

        # 移除最后的 "++"
        formatted_data = formatted_data.rstrip("++") + "\n"

        # 确保目录存在
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.join(out_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(formatted_data)
        print(f"Successfully saved labels to {path}")
        return True
    except Exception as e:
        print(f"Failed to save labels to {path}: {str(e)}")
        return False



# --------- 保存测试数据 ---------
def save_test_data_x(x_data, out_dir, filename="test_data_x.txt"):
    """
    保存测试数据 x 为特定格式
    """
    try:
        formatted_data = ""
        for seq in x_data:
            formatted_data += "++".join([_serialize_array_like(seq[i]) for i in range(seq.shape[0])]) + "\n"

        # 确保目录存在
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.join(out_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(formatted_data)
        print(f"Successfully saved test data x to {path}")
        return True
    except Exception as e:
        print(f"Failed to save test data x to {path}: {str(e)}")
        return False


# --------- 训练和测试数据提取 ---------
def main():
    parser = argparse.ArgumentParser(description="Extract LSTM layers and FC layers and save structure/data")
    parser.add_argument("source", help="Path to python file defining the model")
    parser.add_argument("--out-dir", help="Directory for exported files (default: <source>_export/)")
    parser.add_argument("--include-weights", action="store_true", help="Include weights in the output")
    args = parser.parse_args()

    src = os.path.abspath(args.source)
    if not os.path.isfile(src):
        print("Source file not found:", src)
        sys.exit(2)

    out_dir = args.out_dir or (os.path.splitext(src)[0] + "_export")
    os.makedirs(out_dir, exist_ok=True)

    try:
        module = import_module_from_path(src)
    except Exception as e:
        print(f"Failed to import module: {e}")
        sys.exit(3)

    try:
        model = find_or_build_model(module)
        print("Model loaded successfully:", model)
    except Exception as e:
        print(f"Failed to find or build model: {e}")
        sys.exit(4)

    model.eval()

    # 提取LSTM层
    try:
        lstm_layers = extract_lstm_layers(model, include_weights=args.include_weights)
        print("Extracted LSTM layers:", lstm_layers)
    except Exception as e:
        print(f"Failed to extract LSTM layers: {e}")
        sys.exit(5)

    # 提取全连接层
    try:
        linear_layers = extract_linear_layers(model, include_weights=args.include_weights)
        print("Extracted Linear layers:", linear_layers)
    except Exception as e:
        print(f"Failed to extract Linear layers: {e}")
        sys.exit(6)

    result = {"lstm_layers": lstm_layers, "linear_layers": linear_layers}

    # 保存模型结构到 JSON
    structure_json_path = os.path.join(out_dir, "model_structure.json")
    try:
        with open(structure_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved model structure to {structure_json_path}")
    except Exception as e:
        print(f"Failed to save JSON structure: {e}")

    # 保存LSTM层和全连接层的权重
    save_lstm_weights_to_file(out_dir, lstm_layers)

    for layer in linear_layers:
        save_param_custom(os.path.join(out_dir, f"{layer['name']}_weight.txt"), layer['weight'])
        if 'bias' in layer:
            save_param_custom(os.path.join(out_dir, f"{layer['name']}_bias.txt"), layer['bias'])

    # 提取真实的测试数据 x 和 y
    X_test = torch.randn(20, 100, 13)  # 200 个测试样本
    y_test = torch.randint(0, 10, (20,))  # 200 个测试标签

    # 保存测试数据和标签
    save_test_data_x(X_test, out_dir)
    save_y_labels(y_test, out_dir)


if __name__ == "__main__":
    main()
