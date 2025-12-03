from typing import Optional

class VarNode:
    def __init__(self):
        self.varType: Optional[str] = None       # 变量类型（CPN颜色集）
        self.varName: Optional[str] = None       # 变量名
        self.varIniValue: Optional[str] = None   # 初始值
        self.varBranch: Optional[int] = None     # 分支号
        self.varFor: Optional[bool] = None       # 是否在 for 中声明

    def getVarType(self) -> Optional[str]:
        return self.varType

    def setVarType(self, varType: str):
        self.varType = varType

    def getVarName(self) -> Optional[str]:
        return self.varName

    def setVarName(self, varName: str):
        self.varName = varName

    def getVarIniValue(self) -> Optional[str]:
        return self.varIniValue

    def setVarIniValue(self, varIniValue: str):
        self.varIniValue = varIniValue

    def getVarBranch(self) -> Optional[int]:
        return self.varBranch

    def setVarBranch(self, varBranch: int):
        self.varBranch = varBranch

    def getVarFor(self) -> Optional[bool]:
        return self.varFor

    def setVarFor(self, varFor: bool):
        self.varFor = varFor

    def __str__(self):
        return (
            f"VarNode{{"
            f"varType='{self.varType}', "
            f"varName='{self.varName}', "
            f"varIniValue='{self.varIniValue}', "
            f"varBranch={self.varBranch}, "
            f"varFor={self.varFor}"
            f"}}"
        )
