import random

class BinderNode:
    def __init__(self, instance):
        self.id = random.randint(10000, 10999)
        self.ID = f"ID{self.id}"
        self.panx = -6.0
        self.pany = -5.0
        self.zoom = 1.0
        self.instance = instance

    def set_pan(self, panx, pany):
        self.panx = panx
        self.pany = pany

    def __str__(self):
        return (
            f'          <cpnsheet id="{self.ID}"\n'
            f'                    panx="{self.panx}"\n'
            f'                    pany="{self.pany}"\n'
            f'                    zoom="{self.zoom}"\n'
            f'                    instance="{self.instance}">\n'
            f'            <zorder>\n'
            f'              <position value="0"/>\n'
            f'            </zorder>\n'
            f'          </cpnsheet>\n'
        )
