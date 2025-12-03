from typing import Optional

class MlNode:
    def __init__(self, id_: Optional[str] = None, text: Optional[str] = None):
        self.id = id_
        self.text = text

    def getId(self) -> Optional[str]:
        return self.id

    def setId(self, id_: str):
        self.id = id_

    def getText(self) -> Optional[str]:
        return self.text

    def setText(self, text: str):
        self.text = text

    def __str__(self):
        return (
            f'        <ml id="{self.id}">{self.text}\n'
            f'          <layout>{self.text}</layout>\n'
            f'        </ml>\n'
        )
