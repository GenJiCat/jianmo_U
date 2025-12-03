from typing import Optional

class VarMoNode:
    def __init__(self, type_: Optional[str] = None, name: Optional[str] = None, layout: Optional[str] = None):
        self.id: Optional[str] = None
        self.type: Optional[str] = type_
        self.name: Optional[str] = name
        self.layout: Optional[str] = layout

        if type_ and name:
            self.setLayout()

    def getType(self) -> Optional[str]:
        return self.type

    def setType(self, type_: str):
        self.type = type_

    def getName(self) -> Optional[str]:
        return self.name

    def setName(self, name: str):
        self.name = name
        self.setLayout()

    def getLayout(self) -> Optional[str]:
        return self.layout

    def setLayout(self):
        self.layout = f"var {self.name}:{self.type};"

    def getId(self) -> Optional[str]:
        return self.id

    def setId(self, id_: str):
        self.id = id_

    def setTN(self, type_: str, name: str):
        self.setType(type_)
        self.setName(name)

    def __str__(self):
        return (
            f'        <var id="{self.id}">\n'
            f'          <type>\n'
            f'            <id>{self.type}</id>\n'
            f'          </type>\n'
            f'          <id>{self.name}</id>\n'
            f'          <layout>{self.layout}</layout>\n'
            f'        </var>\n'
        )
