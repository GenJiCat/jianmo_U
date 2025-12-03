import os
import sys
# 将项目根目录加入 Python 模块搜索路径，以便导入 node 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
from createModel import CreatModel
from typing import List, Dict
# 递归提取 assignment 节点
def extract_assignments(node, assignments):
    if not isinstance(node, dict): return
    if node.get("Type") == "assignment":
        exp = node.get("Exp", "")
        if "=" in exp:
            lhs, rhs = [s.strip() for s in exp.split("=", 1)]
            assignments.append((lhs, rhs))
    for child in (node.get("Left"), node.get("Right")):
        if child:
            extract_assignments(child, assignments)

# 主流程
def main(initial, tables_file, tree_file, output_file):
    base_dir = os.path.dirname(os.path.abspath(initial))
    # 读取模板与 JSON
    with open(os.path.join(base_dir, initial), 'r', encoding='utf-8') as f:
        template = [l.rstrip('\n') for l in f]
    tables = json.load(open(os.path.join(base_dir, tables_file), 'r', encoding='utf-8'))
    tree   = json.load(open(os.path.join(base_dir, tree_file),   'r', encoding='utf-8'))

    # 初始化模型
    model = CreatModel()
    model.setCpntxt(template)
    # 添加色集和变量声明
    for var in tables.get('global_vars', []):
        ctype = var['var_type'].upper()
        model.addColor(ctype)
        model.addVar(ctype, var['var_name'])

    # 提取赋值语句
    assignments=[]
    for node in tree.values(): extract_assignments(node, assignments)
    if not assignments: raise RuntimeError('未提取到赋值语句')

    # 构建模型元素
    elements=[]
    var_places = {}
    const_places = {}
    x0, y0, dx = -400.0, 200.0, 200.0
    # 创建目标变量库所（空）
    for idx, (lhs, _) in enumerate(assignments):
        if lhs not in var_places:
            x = x0 + idx*dx
            pstr, pid, px, py = model.addPlace(x+100, y0, 'INT')
            elements.append(pstr)
            var_places[lhs]=pid

    # 为每条赋值创建常量库所、变迁、弧
    for idx, (lhs, rhs) in enumerate(assignments):
        bx, by = x0 + idx*dx, y0-150
        terms = [t.strip() for t in rhs.split('+')]
        # 输入库所
        input_pids=[]
        for term in terms:
            if term.isidentifier():
                pid = var_places.get(term)
            else:
                if term not in const_places:
                    cstr, cid, cx, cy = model.addPlace_init(bx-100, y0, 'INT', initmark=term)
                    elements.append(cstr)
                    const_places[term]=cid
                pid = const_places[term]
            input_pids.append(pid)
        # 变迁
        tstr, tid, tx, ty = model.addTrance(bx, by)

        elements.append(tstr)
        # 输入弧：单常量赋值时标签为变量名lhs，否则为term
        for pid, term in zip(input_pids, terms):
            if not term.isidentifier() and len(terms)==1:
                label = lhs
            else:
                label = term
            astr = model.addArc(tid, pid, 'PtoT', label, (bx-100+bx)/2, (by+float(ty))/2)
            elements.append(astr)
                # 输出弧: 标签为 RHS 表达式或 lhs
        pid_out = var_places[lhs]
        # 单常量赋值时，输出弧也标注为变量名 lhs
        if len(terms) == 1 and not terms[0].isidentifier():
            label_out = lhs
        else:
            label_out = rhs
        astr2 = model.addArc(tid, pid_out, 'TtoP', label_out, bx, (by+float(ty))/2)
        elements.append(astr2)

    # 插入页面并创建实例
    model.addPage(elements, [], True)
    # 写出 CPNXML
    outp = os.path.join(base_dir, output_file)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp,'w',encoding='utf-8') as f:
        f.write('\n'.join(model.getCpntxt()))
    print('Generated:', outp)

def build_assignments(model, assignments, x0=0, y0=200, dx=200):
    """
    根据 (lhs, rhs) 赋值对列表，在当前 model 上生成相应的库所、变迁和弧。
    返回所有新生成元素的 XML 字符串列表，坐标由 x0, y0, dx 控制。
    """
    elems = []
    var_places = {}
    const_places = {}
    # 1. 为每个目标变量创建一个库所
    for idx, (lhs, _) in enumerate(assignments):
        if lhs not in var_places:
            x = x0 + idx * dx
            pstr, pid, px, py = model.addPlace(x + 100, y0, 'INT')
            elems.append(pstr)
            var_places[lhs] = pid

    # 2. 为每条赋值创建常量/变量输入库所、变迁、弧
    for idx, (lhs, rhs) in enumerate(assignments):
        bx = x0 + idx * dx
        by = y0 - 150
        terms = [t.strip() for t in rhs.split('+')]
        # 输入库所
        input_pids = []
        for term in terms:
            if term.isidentifier() and term in var_places:
                pid = var_places[term]
            else:
                # 常量
                if term not in const_places:
                    cstr, cid, cx, cy = model.addPlace_init(bx - 100, y0, 'INT', initmark=term)
                    elems.append(cstr)
                    const_places[term] = cid
                pid = const_places[term]
            input_pids.append(pid)
        # 变迁
        tstr, tid, tx, ty = model.addTrance(bx, by)
        elems.append(tstr)
        # 输入弧
        for pid, term in zip(input_pids, terms):
            label = term if term.isidentifier() or len(terms) > 1 else lhs
            astr = model.addArc(tid, pid, 'PtoT', label, (bx - 100 + bx) / 2, (by + float(ty)) / 2)
            elems.append(astr)
        # 输出弧
        pid_out = var_places[lhs]
        label_out = rhs if len(terms) > 1 or terms[0].isidentifier() else lhs
        astr2 = model.addArc(tid, pid_out, 'TtoP', label_out, bx, (by + float(ty)) / 2)
        elems.append(astr2)

    return elems


if __name__=='__main__':
    p=argparse.ArgumentParser()

    p.add_argument('--initial',default='initial.cpn')
    p.add_argument('--tables', default='tables.json')
    p.add_argument('--tree',   default='tree2.json')
    p.add_argument('--output', default='output/output.cpn')
    args=p.parse_args()
    main(args.initial,args.tables,args.tree,args.output)
