import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
from createModel import CreatModel

# 构建单个子页面
def build_child_page(model: CreatModel, fn: str, assigns: List[Tuple[str,str]], x0: float, y0: float, dx: float) -> Tuple[str, str, str]:
    elems: List[str] = []
    input_pid: str = ''
    output_pid: str = ''
    # 只处理第一个赋值语句
    lhs, rhs = assigns[0]
    parts = [t.strip() for t in rhs.split('+')]
    # 创建 lhs place 和 transition
    pstr, pid_lhs, _, _ = model.addPlace(x0, y0+dx, 'INT')
    elems.append(pstr)
    tstr, tid, _, _ = model.addTrance(x0, y0)
    elems.append(tstr)
    # 输入弧：变量与常量
    for term in parts:
        if term.isidentifier():
            pstr, pid, _, _ = model.addPortPlace(x0, y0, 'INT', 'in')
            elems.append(pstr)
            input_pid = pid
            elems.append(model.addArc(tid, pid, 'PtoT', term, x0, y0))
        else:
            cstr, cid, _, _ = model.addPlace(x0, y0+2*dx, 'INT', False)
            elems.append(cstr)
            elems.append(model.addArc(tid, cid, 'PtoT', term, x0, y0))
    # 输出弧：i+const 或 lhs
    num = next((p for p in parts if not p.isidentifier()), None)
    var = next((p for p in parts if p.isidentifier()), None)
    label_out = f"{var}+{num}" if var and num else lhs
    elems.append(model.addArc(tid, pid_lhs, 'TtoP', label_out, x0, y0))
    output_pid = pid_lhs
    page_pid = model.addPage(elems, [], False)
    return page_pid, input_pid, output_pid

# 提取赋值
def extract_assignments(node: Dict) -> List[Tuple[str,str]]:
    assigns: List[Tuple[str,str]] = []
    if not isinstance(node, dict):
        return assigns
    if node.get('Type') == 'assignment':
        exp = node['Exp']
        lhs, rhs = [s.strip() for s in exp.split('=',1)]
        assigns.append((lhs, rhs))
    return assigns

# 提取调用列表
def extract_child_calls(tree: Dict) -> List[str]:
    calls: List[str] = []
    def walk(n):
        if not isinstance(n, dict): return
        exp = n.get('Exp','')
        if '(' in exp and ')' in exp and not exp.startswith('return'):
            calls.append(exp.split('(')[0].split()[-1])
        for c in ('Left','Right'):
            if n.get(c): walk(n[c])
    walk(tree.get('b',{}))
    return calls

# 主体函数
def main(initial: str, tables_file: str, tree_file: str, output: str):
    with open(initial,'r',encoding='utf-8') as f:
        template = [l.rstrip('\n') for l in f]
    tables = json.load(open(tables_file,'r',encoding='utf-8'))
    tree = json.load(open(tree_file,'r',encoding='utf-8'))

    model = CreatModel()
    model.setCpntxt(template)
    # 添加全局变量
    for var in tables.get('global_vars',[]):
        ctype=var['var_type'].upper(); model.addColor(ctype); model.addVar(ctype,var['var_name'])

    # 获取 b 的子调用
    child_funcs = extract_child_calls(tree)
    child_funcs = list(dict.fromkeys(child_funcs))  # 去重，保持顺序

    # 子页面建模
    page_info: Dict[str, Dict] = {}
    for idx, fn in enumerate(child_funcs):
        assigns = extract_assignments(tree.get(fn,{}))
        page_pid, in_pid, out_pid = build_child_page(model, fn, assigns, x0=100+idx*300, y0=100, dx=80)
        page_info[fn] = {'page_id':page_pid,'in':in_pid,'out':out_pid}

    # 主页面 b
    elems: List[str] = []
    subtrans: List[str] = []
    # 初始化库所 i
    pstr, init_pid, _, _ = model.addPlace_init(100,300,'INT','0')
    elems.append(pstr)
    # 构建替代变迁
    assigns_b = extract_assignments(tree.get('b',{}))
    for idx, fn in enumerate(child_funcs):
        info = page_info[fn]
        lhs = next((l for l,r in assigns_b if r.startswith(f"{fn}(") ), fn)
        # 返回库所
        pstr2, ret_pid, _, _ = model.addPlace(200+idx*300,400,'INT')
        elems.append(pstr2)
        # portsock 双对
        portsock = f"({info['in']},{init_pid})({info['out']},{ret_pid})"
        x=200+idx*300
        tstr, tid, _, _ = model.addSubTrance(x,300,info['page_id'],portsock,fn)
        elems.append(tstr)
        # 输入弧 init->sub
        elems.append(model.addArc(tid,init_pid,'PtoT','i',x-50,300))
        # 输出弧 sub->ret
        elems.append(model.addArc(tid,ret_pid,'TtoP',lhs,x+50,300))
        subtrans.append(tid)
    model.addPage(elems,[subtrans],True)

    # 输出
    os.makedirs(os.path.dirname(output),exist_ok=True)
    with open(output,'w',encoding='utf-8') as f:
        f.write("\n".join(model.getCpntxt()))
    print('Generated:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--initial',default='initial.cpn')
    p.add_argument('--tables',default='aatables.json')
    p.add_argument('--tree',default='tree.json')
    p.add_argument('--output',default='output/output5.cpn')
    args=p.parse_args()
    main(args.initial,args.tables,args.tree,args.output)
