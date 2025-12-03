from typing import Optional

class NotesTable:
    def __init__(self, exp: Optional[str] = None):
        self.varPackage: Optional[str] = None
        self.varClass: Optional[str] = None
        self.varFun: Optional[str] = None
        self.varDeep: Optional[str] = None
        self.varType: Optional[str] = None
        self.varName: Optional[str] = None

        if exp:
            arr = exp.split("-")
            if len(arr) >= 7:
                self.varPackage = arr[1]
                self.varClass = arr[2]
                self.varFun = arr[3]
                self.varDeep = arr[4]
                self.varType = arr[5]
                self.varName = arr[6]

    def getVarPackage(self) -> Optional[str]:
        return self.varPackage

    def setVarPackage(self, varPackage: str):
        self.varPackage = varPackage

    def getVarClass(self) -> Optional[str]:
        return self.varClass

    def setVarClass(self, varClass: str):
        self.varClass = varClass

    def getVarFun(self) -> Optional[str]:
        return self.varFun

    def setVarFun(self, varFun: str):
        self.varFun = varFun

    def getVarDeep(self) -> Optional[str]:
        return self.varDeep

    def setVarDeep(self, varDeep: str):
        self.varDeep = varDeep

    def getVarType(self) -> Optional[str]:
        return self.varType

    def setVarType(self, varType: str):
        self.varType = varType

    def getVarName(self) -> Optional[str]:
        return self.varName

    def setVarName(self, varName: str):
        self.varName = varName

    def __str__(self):
        return (
            f"NotesTable{{varPackage='{self.varPackage}', varClass='{self.varClass}', "
            f"varFun='{self.varFun}', varDeep='{self.varDeep}', "
            f"varType='{self.varType}', varName='{self.varName}'}}"
        )
