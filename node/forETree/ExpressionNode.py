from typing import Optional

from typing import Optional, Any

class ExpressionNode:
    def __init__(self,
                 exp_exp: Optional[str] = None,
                 exp_type: Optional[str] = None,
                 lineno: Optional[int] = None,
                 indent_level: int = 0,
                 scope: Optional[str] = None):
        self.exp_exp = exp_exp
        self.exp_type = exp_type
        self.lineno = lineno
        self.indent_level = indent_level
        self.scope = scope
        self.left: Optional['ExpressionNode'] = None   # 嵌套结构
        self.right: Optional['ExpressionNode'] = None  # 顺序结构

    def to_dict(self):
        return {
            "exp_exp": self.exp_exp,
            "exp_type": self.exp_type,
            "lineno": self.lineno,
            "indent_level": self.indent_level,
            "scope": self.scope,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None
        }



