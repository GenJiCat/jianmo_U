import ast
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import os


@dataclass
class FunctionInfo:
    """函数信息"""
    name: str
    params: List[str]
    body_lines: List[str]
    decorators: List[str]
    line_number: int
    local_vars: List[str]
    is_async: bool = False
    is_static: bool = False
    is_class_method: bool = False


@dataclass
class ClassInfo:
    """类信息"""
    name: str
    bases: List[str]
    body_lines: List[str]
    methods: List[FunctionInfo]
    class_vars: List[str]
    line_number: int


@dataclass
class ImportInfo:
    """导入信息"""
    module: str
    names: List[str]
    alias: str = None
    line_number: int = 0


@dataclass
class VariableInfo:
    """变量信息"""
    name: str
    value: str
    line_number: int
    scope: str  # 'global', 'local', 'constant'
    function_name: str = None


@dataclass
class ModuleInfo:
    """模块信息"""
    filename: str
    module_name: str
    imports: List[ImportInfo]
    global_vars: List[VariableInfo]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    constants: List[VariableInfo]


class PythonModuleParser:
    """Python模块解析器"""

    def __init__(self):
        self.module_info = None
        self.current_function = None
        self.current_class = None

    def parse_file(self, filename: str) -> ModuleInfo:
        """解析Python文件"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件不存在: {filename}")

        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Python语法错误: {e}")

        self.module_info = ModuleInfo(
            filename=filename,
            module_name=os.path.splitext(os.path.basename(filename))[0],
            imports=[],
            global_vars=[],
            functions=[],
            classes=[],
            constants=[]
        )

        # 解析AST
        self._parse_ast(tree)

        return self.module_info

    def _parse_ast(self, tree: ast.AST):
        """解析AST节点 - 只处理模块级别的节点"""
        for node in tree.body:  # 只遍历模块级别的节点
            if isinstance(node, ast.Import):
                self._parse_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._parse_import_from(node)
            elif isinstance(node, ast.FunctionDef):
                self._parse_function(node)
            elif isinstance(node, ast.AsyncFunctionDef):
                self._parse_async_function(node)
            elif isinstance(node, ast.ClassDef):
                self._parse_class(node)
            elif isinstance(node, ast.Assign):
                self._parse_assignment(node)
            elif isinstance(node, ast.AnnAssign):
                self._parse_annotated_assignment(node)

    def _parse_import(self, node: ast.Import):
        """解析import语句"""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                names=[alias.name],
                alias=alias.asname,
                line_number=node.lineno
            )
            self.module_info.imports.append(import_info)

    def _parse_import_from(self, node: ast.ImportFrom):
        """解析from import语句"""
        names = [alias.name for alias in node.names]
        aliases = [alias.asname for alias in node.names]

        for i, name in enumerate(names):
            import_info = ImportInfo(
                module=node.module or "",
                names=[name],
                alias=aliases[i],
                line_number=node.lineno
            )
            self.module_info.imports.append(import_info)

    def _parse_function(self, node: ast.FunctionDef):
        """解析函数定义"""
        # 设置当前函数上下文
        old_function = self.current_function
        self.current_function = node.name

        # 检查是否为类方法
        is_class_method = self._is_in_class_context(node)
        is_static = any(isinstance(d, ast.Name) and d.id == 'staticmethod'
                        for d in node.decorator_list)

        function_info = FunctionInfo(
            name=node.name,
            params=self._extract_params(node.args),
            body_lines=self._extract_body_lines(node.body),
            decorators=self._extract_decorators(node.decorator_list),
            line_number=node.lineno,
            local_vars=[],
            is_async=False,
            is_static=is_static,
            is_class_method=is_class_method
        )

        # 解析函数体中的变量
        self._parse_function_variables(node.body, function_info)

        if is_class_method:
            if self.module_info.classes:
                self.module_info.classes[-1].methods.append(function_info)
        else:
            self.module_info.functions.append(function_info)

        # 恢复函数上下文
        self.current_function = old_function

    def _parse_async_function(self, node: ast.AsyncFunctionDef):
        """解析异步函数定义"""
        old_function = self.current_function
        self.current_function = node.name

        function_info = FunctionInfo(
            name=node.name,
            params=self._extract_params(node.args),
            body_lines=self._extract_body_lines(node.body),
            decorators=self._extract_decorators(node.decorator_list),
            line_number=node.lineno,
            local_vars=[],
            is_async=True,
            is_static=False,
            is_class_method=self._is_in_class_context(node)
        )

        self._parse_function_variables(node.body, function_info)

        if function_info.is_class_method:
            if self.module_info.classes:
                self.module_info.classes[-1].methods.append(function_info)
        else:
            self.module_info.functions.append(function_info)

        self.current_function = old_function

    def _parse_function_variables(self, body: List[ast.stmt], function_info: FunctionInfo):
        """解析函数体中的变量"""
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in function_info.local_vars:
                            function_info.local_vars.append(target.id)
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    if stmt.target.id not in function_info.local_vars:
                        function_info.local_vars.append(stmt.target.id)
            elif isinstance(stmt, ast.If):
                # 递归处理if语句中的变量
                self._parse_function_variables(stmt.body, function_info)
                if stmt.orelse:
                    self._parse_function_variables(stmt.orelse, function_info)

    def _parse_class(self, node: ast.ClassDef):
        """解析类定义"""
        old_class = self.current_class
        self.current_class = node.name

        class_info = ClassInfo(
            name=node.name,
            bases=self._extract_bases(node.bases),
            body_lines=self._extract_body_lines(node.body),
            methods=[],
            class_vars=[],
            line_number=node.lineno
        )

        self.module_info.classes.append(class_info)

        self.current_class = old_class

    def _parse_assignment(self, node: ast.Assign):
        """解析赋值语句 - 只处理模块级别的赋值"""
        # 这里不需要检查 current_function，因为只处理模块级别的节点
        scope = self._determine_variable_scope(node)

        for target in node.targets:
            if isinstance(target, ast.Name):
                var_info = VariableInfo(
                    name=target.id,
                    value=ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value),
                    line_number=node.lineno,
                    scope=scope,
                    function_name=None
                )

                if scope == 'constant':
                    self.module_info.constants.append(var_info)
                elif scope == 'global':
                    self.module_info.global_vars.append(var_info)

    def _parse_annotated_assignment(self, node: ast.AnnAssign):
        """解析带类型注解的赋值 - 只处理模块级别的赋值"""
        if isinstance(node.target, ast.Name):
            scope = self._determine_variable_scope(node)
            var_info = VariableInfo(
                name=node.target.id,
                value=ast.unparse(node.value) if node.value and hasattr(ast, 'unparse') else "",
                line_number=node.lineno,
                scope=scope,
                function_name=None
            )

            if scope == 'global':
                self.module_info.global_vars.append(var_info)

    def _determine_variable_scope(self, node: ast.Assign) -> str:
        """确定变量的作用域"""
        # 检查是否为常量（全大写）
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                return 'constant'

        # 否则为全局变量
        return 'global'

    def _extract_params(self, args: ast.arguments) -> List[str]:
        """提取函数参数"""
        params = []
        for arg in args.args:
            params.append(arg.arg)
        return params

    def _extract_body_lines(self, body: List[ast.stmt]) -> List[str]:
        """提取函数体代码行"""
        lines = []
        for stmt in body:
            if hasattr(ast, 'unparse'):
                lines.append(ast.unparse(stmt))
            else:
                lines.append(str(stmt))
        return lines

    def _extract_decorators(self, decorator_list: List[ast.expr]) -> List[str]:
        """提取装饰器"""
        decorators = []
        for decorator in decorator_list:
            if hasattr(ast, 'unparse'):
                decorators.append(ast.unparse(decorator))
            else:
                decorators.append(str(decorator))
        return decorators

    def _extract_bases(self, bases: List[ast.expr]) -> List[str]:
        """提取类的基类"""
        base_names = []
        for base in bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif hasattr(ast, 'unparse'):
                base_names.append(ast.unparse(base))
            else:
                base_names.append(str(base))
        return base_names

    def _is_in_class_context(self, node: ast.FunctionDef) -> bool:
        """检查函数是否在类上下文中"""
        return self.current_class is not None


def save_module_info(module_info: ModuleInfo, output_file: str):
    """保存模块信息到文件"""
    data = asdict(module_info)

    for func in data['functions']:
        func['body_lines'] = func['body_lines']

    for cls in data['classes']:
        cls['body_lines'] = cls['body_lines']
        for method in cls['methods']:
            method['body_lines'] = method['body_lines']

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """主函数"""
    input_file = "../test/a.py"
    output_file = "../test/aa.json"

    try:
        parser = PythonModuleParser()

        print(f"正在解析文件: {input_file}")
        module_info = parser.parse_file(input_file)

        print(f"正在保存结果到: {output_file}")
        save_module_info(module_info, output_file)

        print("\n解析完成！统计信息:")
        print(f"导入语句: {len(module_info.imports)} 个")
        print(f"全局变量: {len(module_info.global_vars)} 个")
        print(f"常量: {len(module_info.constants)} 个")
        print(f"函数: {len(module_info.functions)} 个")

        for func in module_info.functions:
            print(f"  函数 '{func.name}' 包含 {len(func.local_vars)} 个局部变量: {func.local_vars}")

        print(f"类: {len(module_info.classes)} 个")

        for cls in module_info.classes:
            print(f"  类 '{cls.name}' 包含 {len(cls.methods)} 个方法")

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except SyntaxError as e:
        print(f"语法错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")


if __name__ == "__main__":
    main()