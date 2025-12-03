import ast
import json
from typing import Optional, List, Dict, Any

class ExpressionNode:
    def __init__(
        self,
        exp: str,
        type_: str,
        indent: int = 0,
        branch_num: int = 0,
        granularity: str = "Coarse"
    ):
        self.Exp = exp
        self.Type = type_
        self.Relate = False       # 默认不关联
        self.isFusion = False     # 默认不融合
        self.Indent = indent
        self.BranchNum = branch_num
        self.Granularity = granularity
        self.Left: Optional['ExpressionNode'] = None
        self.Right: Optional['ExpressionNode'] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Exp": self.Exp,
            "Type": self.Type,
            "Relate": self.Relate,
            "isFusion": self.isFusion,
            "Indent": self.Indent,
            "BranchNum": self.BranchNum,
            "Granularity": self.Granularity,
            "Left": self.Left.to_dict() if self.Left else None,
            "Right": self.Right.to_dict() if self.Right else None
        }

# 映射 ast 节点到自定义类型标签
AST_TYPE_MAP = {
    ast.Assign: "assignment",
    ast.AugAssign: "assignment",
    ast.If: "if",
    ast.For: "for",
    ast.While: "while",
    ast.FunctionDef: "def",
    ast.ClassDef: "class",
    ast.Import: "import",
    ast.ImportFrom: "importfrom",
    ast.Expr: "expr",
    ast.Return: "return",
    ast.With: "with",
    ast.Try: "try"
}

def get_type(node: ast.AST) -> str:
    return AST_TYPE_MAP.get(type(node), type(node).__name__.lower())

# 从源代码行提取文本
def get_code_line(lines: List[str], node: ast.AST) -> str:
    if hasattr(node, 'lineno'):
        return lines[node.lineno - 1].strip()
    return ''

# 将 AST 转为 ExpressionNode 列表
def ast_to_nodes(
    tree: ast.AST,
    source_lines: List[str],
    indent: int = 4
) -> List[ExpressionNode]:
    nodes: List[ExpressionNode] = []
    for node in ast.iter_child_nodes(tree):
        exp_text = get_code_line(source_lines, node)
        node_type = get_type(node)
        exp_node = ExpressionNode(
            exp=exp_text,
            type_=node_type,
            indent=indent
        )
        # 递归处理嵌套代码块
        if hasattr(node, 'body') and isinstance(node.body, list) and node.body:
            children = ast_to_nodes(
                ast.Module(body=node.body),
                source_lines,
                indent=indent + 4
            )
            exp_node.Left = build_tree(children)
        # 处理 orelse 分支
        if hasattr(node, 'orelse') and isinstance(node.orelse, list) and node.orelse:
            orelse_children = ast_to_nodes(
                ast.Module(body=node.orelse),
                source_lines,
                indent=indent + 4
            )
            exp_node.Right = build_tree(orelse_children)
        nodes.append(exp_node)
    return nodes

# 将列表连成二叉树: 顺序用 Right 链接
def build_tree(nodes: List[ExpressionNode]) -> Optional[ExpressionNode]:
    if not nodes:
        return None
    root = nodes[0]
    curr = root
    for nxt in nodes[1:]:
        curr.Right = nxt
        curr = nxt
    return root

# 主流程: 读取源文件与变量表，并生成按函数命名的语义二叉树
def generate_semantic_tree(
    python_path: str,
    vars_json_path: str,
    output_json_path: str
):
    # 读取源代码
    with open(python_path, 'r', encoding='utf-8') as f:
        source = f.read()
    lines = source.splitlines()
    module = ast.parse(source, filename=python_path)
    # 读取变量表
    with open(vars_json_path, 'r', encoding='utf-8') as f:
        var_table = json.load(f)
    # 构建结果字典，以函数名为键
    result: Dict[str, Any] = {}
    # 遍历模块顶层函数定义
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # 只处理函数体
            children = ast_to_nodes(
                ast.Module(body=node.body),
                lines,
                indent=4
            )
            tree_root = build_tree(children)
            # TODO: 根据 var_table 标记节点的 Relate/isFusion
            result[func_name] = tree_root.to_dict() if tree_root else {}
    # 输出到 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"语义二叉树已保存到 {output_json_path}")

if __name__ == '__main__':
    pyfile = input("请输入源 .py 文件路径：").strip()
    varfile = input("请输入变量表 JSON 文件路径：").strip()
    outfile = input("请输入输出 JSON 文件路径：").strip()
    generate_semantic_tree(pyfile, varfile, outfile)
