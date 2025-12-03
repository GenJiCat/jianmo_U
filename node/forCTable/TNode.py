from typing import List, Optional

class TableNode:
    def __init__(self, msgs: Optional[str] = None, name: Optional[str] = None):
        self.Msgs = msgs  # 融合集序号
        self.Name = name  # 变量或函数名

    def get_msgs(self) -> Optional[str]:
        return self.Msgs

    def set_msgs(self, msgs: str):
        self.Msgs = msgs

    def get_name(self) -> Optional[str]:
        return self.Name

    def set_name(self, name: str):
        self.Name = name

    def __str__(self):
        return f"TableNode{{Msgs='{self.Msgs}', Name='{self.Name}'}}"


class FunNode(TableNode):
    def __init__(self, exp: Optional[List[str]] = None, msgs: Optional[str] = None, name: Optional[str] = None):
        super().__init__(msgs, name)
        self.Exp = exp if exp is not None else []

    def get_exp(self) -> List[str]:
        return self.Exp

    def set_exp(self, exp: List[str]):
        self.Exp = exp

    def __str__(self):
        return f"FunNode{{Exp='{self.Exp}', Msgs='{self.Msgs}', Name='{self.Name}'}}"
class SvariableNode(TableNode):
    def __init__(self, exp: Optional[str] = None, msgs: Optional[str] = None,
                 name: Optional[str] = None, var_type: Optional[str] = None):
        super().__init__(msgs, name)
        self.Exp = exp  # 声明语句
        self.Type = var_type  # 变量类型

    def get_type(self) -> Optional[str]:
        return self.Type

    def set_type(self, var_type: str):
        self.Type = var_type

    def get_exp(self) -> Optional[str]:
        return self.Exp

    def set_exp(self, exp: str):
        self.Exp = exp

    def __str__(self):
        return f"SvariableNode{{Exp='{self.Exp}', Msgs='{self.Msgs}', Type='{self.Type}', Name='{self.Name}'}}"
class VariableNode(TableNode):
    def __init__(
        self,
        exp: Optional[str] = None,
        msgs: Optional[str] = None,
        name: Optional[str] = None,
        var_type: Optional[str] = None
    ):
        super().__init__(msgs, name)
        self.Exp = exp  # 声明语句
        self.Type = var_type  # 变量类型

    @classmethod
    def from_type(cls, var_type: str):
        return cls(var_type=var_type)

    def get_type(self) -> Optional[str]:
        return self.Type

    def set_type(self, var_type: str):
        self.Type = var_type

    def get_exp(self) -> Optional[str]:
        return self.Exp

    def set_exp(self, exp: str):
        self.Exp = exp

    def __str__(self):
        return f"VariableNode{{Exp='{self.Exp}', Msgs='{self.Msgs}', Type='{self.Type}', Name='{self.Name}'}}"