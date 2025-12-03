import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
from createModel import CreatModel

# 递归提取 loop 节点
def extract_loops(node, loops):
    if not isinstance(node, dict):
        return
    if node.get("Type") in ("while", "for"):
        loops.append(node)
    for child in (node.get("Left"), node.get("Right")):
        if child:
            extract_loops(child, loops)

# 在树中找到变量初始值和增量
def find_init_and_increment(var_name, tree):
    init_val = None
    incr_val = 1

    def search(node):
        nonlocal init_val, incr_val
        if not isinstance(node, dict):
            return

        exp = node.get("Exp", "").replace(" ", "")
        typ = node.get("Type")

        if typ == "assignment":
            if exp.startswith(f"{var_name}="):
                rhs = exp.split("=")[1]
                # 初始值 i=10
                if rhs.isdigit():
                    init_val = int(rhs)
                # 增量值 i=i+10
                elif rhs.startswith(f"{var_name}+"):
                    try:
                        incr_val = int(rhs.split("+")[1])
                    except:
                        pass
        elif typ == "for":
            # 处理 for a in (3,20):
            if exp.startswith(f"for{var_name}in("):
                try:
                    range_str = exp.split("in(")[1].rstrip("):")
                    init_val = int(range_str.split(",")[0])
                except:
                    pass

        for k in ("Left", "Right"):
            if node.get(k):
                search(node[k])

    for node in tree.values():
        search(node)

    return (init_val if init_val is not None else 0), incr_val

# XML 特殊字符转义
def xml_escape(s: str) -> str:
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;')
             .replace("'", '&apos;'))

# 主流程
def main(initial, tables_file, tree_file, output_file):
    base_dir = os.path.dirname(os.path.abspath(initial))
    # 读取模板与 JSON
    with open(os.path.join(base_dir, initial), 'r', encoding='utf-8') as f:
        template = [l.rstrip('\n') for l in f]
    with open(os.path.join(base_dir, tables_file), 'r', encoding='utf-8') as f:
        tables = json.load(f)
    with open(os.path.join(base_dir, tree_file), 'r', encoding='utf-8') as f:
        tree = json.load(f)

    # 初始化模型
    model = CreatModel()
    model.setCpntxt(template)
    # 添加色集和变量声明
    for var in tables.get('global_vars', []):
        ctype = var['var_type'].upper()
        model.addColor(ctype)
        model.addVar(ctype, var['var_name'])

    # 提取循环语句
    loops = []
    for node in tree.values():
        extract_loops(node, loops)
    if not loops:
        raise RuntimeError('未提取到循环语句')

    elements = []
    x0, y0, dx, dy = -400.0, 200.0, 300.0, 200.0
    # 对每个循环生成库所和转移
    for idx, loop in enumerate(loops):
        exp = loop['Exp'].rstrip(':').strip()  # e.g. "while i<10" or "for a in (3,20)"
        if loop['Type'] == 'while':
            raw_cond = exp.split(None, 1)[1]  # "i<10"
            var_name = raw_cond.split('<')[0].strip()
            cond_expr = xml_escape(raw_cond)
        else:
            # for a in (3,20) -> a < 20
            parts = exp.split()
            var_name = parts[1]
            cond_expr = xml_escape(f"{var_name} < {parts[3].strip('():').split(',')[1]}")

        # 获取初始值和增量
        init_val, incr_val = find_init_and_increment(var_name, tree)
        # print(cond_expr)
        # 坐标
        px = x0 + idx * dx
        py = y0
        # 创建两个库所：循环条件检查库所 input_place 和循环后库所 exit_place
        p1_str, p1_id, _, _ = model.addPlace_init(px, py, 'INT', initmark=init_val)
        p2_str, p2_id, _, _ = model.addPlace(px + dx/2, py, 'INT')
        elements.extend([p1_str, p2_str])

        # 进入循环转移
        _, t1_id, _, _ = model.addTrance(px, py - dy)
        elements.append(model.addArc(t1_id, p1_id, 'PtoT', var_name, px, py - dy/2))
        node1 = model.getTransNode(t1_id)
        node1.set_cond()
        node1.cond.text = cond_expr
        node1.set_time()
        node1.set_code()
        node1.set_priority()
        # 输出弧：从 t1 回到 p1，label 为变量自增表达式
        incr_label = f"{var_name}+{incr_val}"
        elements.append(model.addArc(t1_id, p1_id, 'TtoP', xml_escape(incr_label), px, py - dy/2))
        elements.append(node1.__str__())

        # 退出循环转移
        _, t2_id, _, _ = model.addTrance(px + dx/4, py - dy)
        elements.append(model.addArc(t2_id, p1_id, 'PtoT', var_name, px + dx/4, py - dy/2))
        node2 = model.getTransNode(t2_id)
        node2.set_cond()
        node2.cond.text = f"not({cond_expr})"
        node2.set_time()
        node2.set_code()
        node2.set_priority()
        elements.append(model.addArc(t2_id, p2_id, 'TtoP', var_name, px + dx/4, py - dy/2))
        elements.append(node2.__str__())

    # 插入页面并创建实例
    model.addPage(elements, [], True)

    # 写出 CPNXML
    outp = os.path.join(base_dir, output_file)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(model.getCpntxt()))
    print('Generated:', outp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', default='initial.cpn')
    parser.add_argument('--tables', default='btables.json')
    parser.add_argument('--tree', default='btree.json')
    parser.add_argument('--output', default='output/output3.cpn')
    args = parser.parse_args()
    main(args.initial, args.tables, args.tree, args.output)
