import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field
import os
import ast
import re


@dataclass
class PythonVarNode:
    """Python变量节点 - 对应Java的VarNode"""
    var_name: str
    var_type: str = "any"  # Python动态类型，默认为any
    var_ini_value: str = ""
    var_branch: int = 0
    var_for: bool = False
    var_scope: str = "local"  # local, global, module
    function_name: str = None


@dataclass
class PythonFunNode:
    """Python函数节点 - 对应Java的FunNode"""
    name: str
    exp: List[str] = field(default_factory=list)  # 函数体代码行
    msgs: str = ""  # 页面标识
    params: List[str] = field(default_factory=list)
    is_async: bool = False
    is_static: bool = False
    is_class_method: bool = False
    local_vars: List[str] = field(default_factory=list)


@dataclass
class PythonClassNode:
    """Python类节点 - 对应Java的ClassNode"""
    name: str
    exp: List[str] = field(default_factory=list)  # 类体代码行
    msgs: str = ""  # 页面标识
    bases: List[str] = field(default_factory=list)
    methods: List[PythonFunNode] = field(default_factory=list)
    class_vars: List[str] = field(default_factory=list)


@dataclass
class PythonModuleTable:
    """Python模块表 - 对应Java的ClassAllTable"""
    filename: str
    module_name: str = ""
    imports: List[str] = field(default_factory=list)
    global_vars: List[PythonVarNode] = field(default_factory=list)
    constants: List[PythonVarNode] = field(default_factory=list)
    functions: List[PythonFunNode] = field(default_factory=list)
    classes: List[PythonClassNode] = field(default_factory=list)
    all_function_names: List[str] = field(default_factory=list)
    all_class_names: List[str] = field(default_factory=list)
    all_global_var_names: List[str] = field(default_factory=list)


def infer_type_from_value(value_str: str) -> str:
    """根据初始值推断变量类型"""
    if not value_str or value_str.strip() == "":
        return "any"

    value_str = value_str.strip()

    # 尝试解析为整数
    try:
        int(value_str)
        return "int"
    except ValueError:
        pass

    # 尝试解析为浮点数
    try:
        float(value_str)
        return "float"
    except ValueError:
        pass

    # 字符串类型
    if (value_str.startswith('"') and value_str.endswith('"')) or \
            (value_str.startswith("'") and value_str.endswith("'")):
        return "str"

    # 布尔类型
    if value_str.lower() in ['true', 'false']:
        return "bool"

    # 列表类型
    if value_str.startswith('[') and value_str.endswith(']'):
        return "list"

    # 字典类型
    if value_str.startswith('{') and value_str.endswith('}'):
        return "dict"

    # 元组类型
    if value_str.startswith('(') and value_str.endswith(')'):
        return "tuple"

    # 集合类型
    if value_str.startswith('{') and value_str.endswith('}') and ':' not in value_str:
        return "set"

    # None类型
    if value_str.lower() == 'none':
        return "None"

    # 函数调用或其他表达式
    if '(' in value_str and ')' in value_str:
        return "function_call"

    # 变量引用
    if value_str.isidentifier():
        return "variable_ref"

    return "any"


def extract_var_assignments(body_lines: List[str]) -> Dict[str, str]:
    """提取函数体中的变量赋值，只保留第一次赋值，跳过增强赋值"""
    assignments = {}

    for line in body_lines:
        line = line.strip()
        # 跳过注释行
        if line.startswith('#'):
            continue

        # 跳过增强赋值，让它们使用已有上下文来推断类型
        if any(op in line for op in ['+=', '-=', '*=', '/=', '//=', '%=', '**=']):
            continue

        # 只处理简单的赋值语句
        if '=' in line:
            var_part, val_part = line.split('=', 1)
            var_name = var_part.strip()
            value = val_part.strip().rstrip(';')

            if var_name.isidentifier() and var_name not in assignments:
                assignments[var_name] = value

    return assignments



def is_for_loop_variable(var_name: str, body_lines: List[str]) -> bool:
    """判断是否为for循环变量"""
    for line in body_lines:
        line = line.strip()
        if line.startswith('for ') and var_name in line:
            return True
    return False


def infer_type_from_expression(expr: str, var_context: Dict[str, str]) -> str:
    """根据表达式和变量上下文推断类型"""
    expr = expr.strip()

    # 如果表达式为空，返回any
    if not expr:
        return "any"

    # 检查是否是简单的字面量
    literal_type = infer_type_from_value(expr)
    if literal_type != "any":
        return literal_type

    # 检查是否是变量引用
    if expr.isidentifier():
        return var_context.get(expr, "any")

    # 检查是否是函数调用
    if '(' in expr and ')' in expr:
        return "function_call"

    # 分析表达式中的操作符和变量
    return analyze_expression_type(expr, var_context)


def analyze_expression_type(expr: str, var_context: Dict[str, str]) -> str:
    """分析表达式的类型"""
    expr = expr.strip()

    # 算术运算（+、-、*、/、//、%、**）
    arithmetic_ops = ['+', '-', '*', '/', '//', '%', '**']
    if any(op in expr for op in arithmetic_ops):
        # 检查操作数类型
        operands = extract_operands(expr, arithmetic_ops)
        operand_types = []

        for operand in operands:
            operand = operand.strip()
            if operand.isidentifier():
                operand_types.append(var_context.get(operand, "any"))
            else:
                operand_types.append(infer_type_from_value(operand))

        # 根据操作数类型推断结果类型
        return infer_arithmetic_result_type(operand_types)

    # 比较运算（==、!=、<、>、<=、>=）
    comparison_ops = ['==', '!=', '<', '>', '<=', '>=']
    if any(op in expr for op in comparison_ops):
        return "bool"

    # 逻辑运算（and、or、not）
    logical_ops = ['and', 'or', 'not']
    if any(op in expr for op in logical_ops):
        return "bool"

    # 字符串操作
    if '+' in expr and ('"' in expr or "'" in expr):
        return "str"

    # 列表操作
    if '[' in expr and ']' in expr:
        return "list"

    return "any"


def extract_operands(expr: str, operators: List[str]) -> List[str]:
    """提取表达式中的操作数"""
    # 简单的操作数提取（可以进一步优化）
    for op in operators:
        if op in expr:
            parts = expr.split(op)
            return [part.strip() for part in parts if part.strip()]
    return [expr]


def infer_arithmetic_result_type(operand_types: List[str]) -> str:
    """根据操作数类型推断算术运算结果类型"""
    if not operand_types:
        return "any"

    # 如果所有操作数都是数字类型
    numeric_types = ['int', 'float']
    if all(t in numeric_types for t in operand_types):
        # 如果有float，结果就是float
        if 'float' in operand_types:
            return "float"
        else:
            return "int"

    # 如果包含字符串，可能是字符串拼接
    if 'str' in operand_types:
        return "str"

    return "any"


def build_variable_table(module_data: Dict) -> List[PythonVarNode]:
    """构建变量表 - 使用上下文信息进行精确类型推断"""
    var_list = []
    var_context = {}  # 变量名到类型的映射

    # 处理全局变量
    for var_info in module_data.get('global_vars', []):
        var_type = infer_type_from_value(var_info.get('value', ''))
        var_node = PythonVarNode(
            var_name=var_info['name'],
            var_type=var_type,
            var_ini_value=var_info.get('value', ''),
            var_branch=0,
            var_for=False,
            var_scope='global',
            function_name=None
        )
        var_list.append(var_node)
        var_context[var_info['name']] = var_type

    # 处理函数中的局部变量
    for func_info in module_data.get('functions', []):
        func_name = func_info['name']
        local_vars = func_info.get('local_vars', [])
        func_body = func_info.get('body_lines', [])

        # 提取函数中的变量赋值
        var_assignments = extract_var_assignments(func_body)

        # 按顺序处理变量，建立上下文
        func_var_context = var_context.copy()  # 继承全局上下文

        # 首先处理有初始值的变量
        for var_name in local_vars:
            if var_assignments.get(var_name, '') != '':
               assignment = var_assignments.get(var_name, '')
            else:
                assignment = 0

            # 检查是否为for循环变量
            is_for_var = is_for_loop_variable(var_name, func_body)

            if assignment:
                # 尝试从赋值语句推断类型
                var_type = infer_type_from_expression(assignment, func_var_context)
            elif is_for_var:
                # for循环变量默认为int类型
                var_type = "int"
            else:
                var_type = "any"

            var_node = PythonVarNode(
                var_name=var_name,
                var_type=var_type,
                var_ini_value=assignment,
                var_branch=1,
                var_for=is_for_var,
                var_scope='local',
                function_name=func_name
            )
            var_list.append(var_node)

            # 更新上下文
            func_var_context[var_name] = var_type

    return var_list


def build_function_table(module_data: Dict) -> List[PythonFunNode]:
    """构建函数表"""
    fun_list = []

    for func_info in module_data.get('functions', []):
        fun_node = PythonFunNode(
            name=func_info['name'],
            exp=func_info.get('body_lines', []),
            msgs=f"function_{func_info['name']}",
            params=func_info.get('params', []),
            is_async=func_info.get('is_async', False),
            is_static=func_info.get('is_static', False),
            is_class_method=func_info.get('is_class_method', False),
            local_vars=func_info.get('local_vars', [])
        )
        fun_list.append(fun_node)

    return fun_list


def build_class_table(module_data: Dict) -> List[PythonClassNode]:
    """构建类表"""
    class_list = []

    for class_info in module_data.get('classes', []):
        class_node = PythonClassNode(
            name=class_info['name'],
            exp=class_info.get('body_lines', []),
            msgs=f"class_{class_info['name']}",
            bases=class_info.get('bases', []),
            methods=[],  # 需要进一步处理类中的方法
            class_vars=class_info.get('class_vars', [])
        )
        class_list.append(class_node)

    return class_list


def build_all_tables(json_file_path: str) -> PythonModuleTable:
    """构建所有表结构"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        module_data = json.load(f)

    # 构建各种表
    var_list = build_variable_table(module_data)
    fun_list = build_function_table(module_data)
    class_list = build_class_table(module_data)

    # 提取所有名称列表
    all_function_names = [func['name'] for func in module_data.get('functions', [])]
    all_class_names = [cls['name'] for cls in module_data.get('classes', [])]
    all_global_var_names = [var['name'] for var in module_data.get('global_vars', [])]

    # 创建模块表
    module_table = PythonModuleTable(
        filename=module_data.get('filename', ''),
        module_name=module_data.get('module_name', ''),
        imports=module_data.get('imports', []),
        global_vars=var_list,
        constants=module_data.get('constants', []),
        functions=fun_list,
        classes=class_list,
        all_function_names=all_function_names,
        all_class_names=all_class_names,
        all_global_var_names=all_global_var_names
    )

    return module_table


def save_tables_to_json(module_table: PythonModuleTable, output_file: str):
    """将表结构保存到JSON文件"""
    # 将dataclass转换为字典
    table_dict = asdict(module_table)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(table_dict, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    input_file = "../test/aa.json"
    output_file = "aatables.json"

    # 构建所有表
    module_table = build_all_tables(input_file)

    # 保存到文件
    save_tables_to_json(module_table, output_file)

    # 打印统计信息
    print("表结构构建完成！")
    print(f"变量表: {len(module_table.global_vars) if module_table.global_vars else 0} 个变量")
    print(f"函数表: {len(module_table.functions) if module_table.functions else 0} 个函数")
    print(f"类表: {len(module_table.classes) if module_table.classes else 0} 个类")

    # 打印变量类型统计
    type_count = {}
    if module_table.global_vars:
        for var in module_table.global_vars:
            type_count[var.var_type] = type_count.get(var.var_type, 0) + 1

    print("\n变量类型统计:")
    for var_type, count in type_count.items():
        print(f"  {var_type}: {count} 个")

    # 打印详细的变量信息
    print("\n详细变量信息:")
    if module_table.global_vars:
        for var in module_table.global_vars:
            print(f"  {var.var_name}: {var.var_type} = {var.var_ini_value} (作用域: {var.var_scope})")


if __name__ == "__main__":
    main()