# expression_array.py
from param import List

from maintool import ExpressionTree
from node.forETree import ExpressionNode
from node.forETree import NotesTable

class ExpressionArray:
    """
    把输入的 Python 源码（按行切分的字符串列表）转换成 ExpressionNode 列表，
    处理注释、判断语句类型、打标细/粗粒度等。
    """
    def __init__(self):
        self.expression_list = []
        self.notes_table = NotesTable()  # 全局注释表
        self.fundecla = None  # 函数定义行

    def save_exp_array(self, lines: list[str]) -> list[ExpressionNode]:
        """
        lines: 源码每行，没有尾部换行符。
        第 0 行当作函数定义；其余行若不是注释行，就创建 ExpressionNode。
        支持：# 单行注释（当作“局域注释”或“广域注释”）。
        """
        if not lines:
            return []

        self.fundecla = lines[0]
        self.expression_list = []

        # 先处理全局注释表
        self.create_notes_table(lines)

        for idx, line in enumerate(lines[1:], start=1):
            stripped = line.strip()
            # 如果是 Python 注释行，以 "#" 开头，当作“注释”，跳过生成节点
            if stripped.startswith("#"):
                continue

            node = ExpressionNode()
            node.exp_exp = line  # 原始语句
            self.judgment_statement_type(node)
            self.judgment_exp_relate(node)

            # 局域注释：上一行若是注释，则视为与本行相关
            prev = lines[idx - 1].strip()
            if prev.startswith("#"):
                # 注释格式：# key-xxx  （用 "-" 分割，取 key）
                parts = prev.lstrip("#").split("-", 1)
                if len(parts) == 2 and parts[1] in node.exp_exp:
                    node.exp_relate = True

            if node.exp_relate:
                node.exp_granularity = "Fine"

            self.expression_list.append(node)

        return self.expression_list

    def create_notes_table(self, lines: list[str]):
        """如果第一行是注释，构建全局 NotesTable"""
        first = lines[0].strip()
        if first.startswith("#"):
            # 假设 NotesTable 接受注释行（不含 "#"）初始化
            self.notes_table = NotesTable(first.lstrip("#").strip())

    def judgment_exp_relate(self, node: ExpressionNode):
        """判断与全局注释的变量名是否相关"""
        varname = self.notes_table.var_name
        if varname and node.exp_type == "assignment" and varname in node.exp_exp:
            node.exp_relate = True

    def judgment_statement_type(node):
        """
        根据 Python 语法特征为 ExpressionNode.exp_type 赋值。
        支持的类型包括：
          - def       函数定义
          - class     类定义
          - import    import / from … import
          - assignment 赋值语句（单个 =，排除 ==）
          - for       for 循环
          - while     while 循环
          - if        if 分支
          - elif      elif 分支
          - else      else 分支
          - try       try
          - except    except
          - finally   finally
          - with      with 上下文管理
          - return    return
          - break     break
          - continue  continue
          - pass      pass
          - raise     raise
          - yield     yield
          - async     async/await
          - other     其它
        """
        exp = node.exp_exp.strip()

        # 函数和类定义
        if exp.startswith("def "):
            node.exp_type = "def"
            return
        if exp.startswith("class "):
            node.exp_type = "class"
            return

        # 导入语句
        if exp.startswith("import ") or exp.startswith("from "):
            node.exp_type = "import"
            return

        # 控制流结构
        if exp.startswith("for "):
            node.exp_type = "for"
            return
        if exp.startswith("while "):
            node.exp_type = "while"
            return
        if exp.startswith("if "):
            node.exp_type = "if"
            return
        if exp.startswith("elif "):
            node.exp_type = "elif"
            return
        if exp.startswith("else"):
            node.exp_type = "else"
            return
        if exp.startswith("try"):
            node.exp_type = "try"
            return
        if exp.startswith("except"):
            node.exp_type = "except"
            return
        if exp.startswith("finally"):
            node.exp_type = "finally"
            return
        if exp.startswith("with "):
            node.exp_type = "with"
            return

        # 异步相关
        if exp.startswith("async "):
            node.exp_type = "async"
            return
        if "await " in exp:
            node.exp_type = "await"
            return

        # 跳转与结束
        if exp.startswith("return "):
            node.exp_type = "return"
            return
        if exp == "return":
            node.exp_type = "return"
            return
        if exp == "break":
            node.exp_type = "break"
            return
        if exp == "continue":
            node.exp_type = "continue"
            return
        if exp == "pass":
            node.exp_type = "pass"
            return
        if exp.startswith("raise "):
            node.exp_type = "raise"
            return
        if exp.startswith("yield "):
            node.exp_type = "yield"
            return

        # 赋值（排除比较 ==、>=、<=、!=）
        # 匹配单个 =，两边不紧邻 = 则当作赋值
        if "=" in exp and not any(op in exp for op in (
        "==", "<=", ">=", "!=", "+=", "-=", "*=", "/=", "//=", "%=", "**=")) and not exp.endswith(":"):
            node.exp_type = "assignment"
            return

        # 其它语句
        node.exp_type = "other"

    def exp_tree(self, exp_list: List[ExpressionNode]) -> ExpressionTree:
        """
        构造并返回表达式二叉树：
        - 如果有语句列表，调用 creat_exp_tree 并传入函数声明；
        - 否则根据 fundecla 判断是抽象函数还是接口函数，打印提示。
        """
        self.expression_list = exp_list
        tree = ExpressionTree()

        if self.expression_list:
            # 构建二叉树，并把函数声明传给树
            tree.creat_exp_tree(self.expression_list)
            tree.set_fundecla(self.fundecla)
            # 如果需要二次类型标注，可以在这里调用
            # tree.statement_type()
        else:
            # 无语句体
            if self.fundecla and "abstract" in self.fundecla:
                print("此函数为抽象函数，无函数体。")
                print(f"函数声明语句：{self.fundecla}")
            else:
                print("此函数为接口内函数，无函数体。")
                print(f"函数声明语句：{self.fundecla}")

        return tree
