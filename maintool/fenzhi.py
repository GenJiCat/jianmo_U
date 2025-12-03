import os
import json
import argparse
from createModel import CreatModel

def extract_branches(node, branches):
    """递归搜集所有 if/elif/else 节点"""
    if not isinstance(node, dict):
        return
    exp = node.get("Exp", "").strip()
    # 收集 if/elif
    if node.get("Type") == "if":
        branches.append(node)
    # 收集 else 分支：Exp 以 else 开头
    elif exp.startswith("else"):
        branches.append(node)
    for child in (node.get("Left"), node.get("Right")):
        if child:
            extract_branches(child, branches)


def xml_escape(s: str) -> str:
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;'))


def main(initial, tables, tree, output):
    # 1. 读入模板与数据
    with open(initial, encoding='utf-8') as f:
        template = [l.rstrip("\n") for l in f]
    tables = json.load(open(tables, encoding='utf-8'))
    tree   = json.load(open(tree,   encoding='utf-8'))

    # 2. 新建模型
    model = CreatModel()
    model.setCpntxt(template)

    # 3. 全局变量 place 声明 & 初始化
    elements = []
    x0, y0, dx, dy = -300, 200, 200, 150
    entry_p_id = None
    for idx, var in enumerate(tables['global_vars']):
        name  = var['var_name']
        vtype = var['var_type'].upper()
        initv = var['var_ini_value']
        model.addColor(vtype)
        model.addVar(vtype, name)
        # 使用 addPlace_init 直接设置初始 marking
        p_str, p_id, _, _ = model.addPlace_init(x0 + idx*dx, y0, vtype, initv)
        elements.append(p_str + "\n")
        if idx == 0:
            entry_p_id = p_id

    # 4. 提取所有分支节点
    branches = []
    for fn_node in tree.values():
        extract_branches(fn_node, branches)
    if not branches:
        raise RuntimeError("没有找到任何 if/else 语句")

    # 5. 构建条件列表，以便处理 else 分支
    conds = []
    for br in branches:
        exp = br.get('Exp', '').rstrip(':').strip()
        if exp.startswith('if') or exp.startswith('elif'):
            conds.append(exp.split(None, 1)[1])
        else:
            conds.append(None)

    # 6. 为每个分支生成变迁和弧
    for idx, br in enumerate(branches):
        exp = br.get('Exp', '').rstrip(':').strip()
        # Guard 条件
        if conds[idx] is not None:
            cond = conds[idx]
        else:
            nots = [f"(not({c}))" for c in conds[:idx] if c]
            cond = ' andalso '.join(nots) if nots else 'true'

        tx = x0 + idx*dx
        ty = y0 - dy
        # 添加变迁
        t_str, t_id, _, _ = model.addTrance(tx, ty)
        node = model.getTransNode(t_id)
        node.set_cond()
        node.cond.text = xml_escape(cond)
        node.set_time(); node.set_code(); node.set_priority()
        br_trans = node.__str__()

        # 输入弧：entry -> 变迁
        elements.append(model.addArc(t_id, entry_p_id, "PtoT", "a", tx, ty+dy/2))

        # 输出 place
        out_px, out_py = tx, y0
        p_out_str, p_out_id, _, _ = model.addPlace(out_px, out_py, "INT")
        elements.append(p_out_str + "\n")

        # 输出弧：变迁 -> place，带赋值 RHS
        # 区分 if/elif 与 else
        rhs = ''
        if exp.startswith('if') or exp.startswith('elif'):
            assign_node = br.get('Left') or {}
        else:
            assign_node = br
        assign_exp = assign_node.get('Exp', '')
        if '=' in assign_exp:
            rhs = assign_exp.split('=', 1)[1]
        label = xml_escape(rhs)
        elements.append(model.addArc(t_id, p_out_id, "TtoP", label, tx, ty+dy/2))

        # 添加变迁 XML
        elements.append(br_trans + "\n")

    # 7. 合并页面并输出 XML
    model.addPage(elements, [], instance=True)
    xml = "\n".join(model.getCpntxt())
    with open(output, 'w', encoding='utf-8') as f:
        f.write(xml)
    print("生成完毕：", output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--initial", default="initial.cpn")
    p.add_argument("--tables",  default="ctables.json")
    p.add_argument("--tree",    default="ctrees.json")
    p.add_argument("--output",  default="output/branch.cpn")
    main(**vars(p.parse_args()))
