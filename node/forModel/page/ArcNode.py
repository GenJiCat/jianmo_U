import random
from node.forModel.page.BasicNode import BasicNode

class ArcNode(BasicNode):
    class Arrowattr:
        def __init__(self):
            self.headsize = 1.2
            self.currentcyckle = 2

        def __str__(self):
            return (
                f'        <arrowattr headsize="{self.headsize}"\n'
                f'                   currentcyckle="{self.currentcyckle}"/>\n'
            )

    class Bendpoint:
        def __init__(self, id_):
            self.id = id_
            self.serial = None
            self.posattr = BasicNode.Posattr()
            self.fillattr = BasicNode.Fillattr()
            self.lineattr = BasicNode.Lineattr()
            self.textattr = BasicNode.Textattr()

        def __str__(self):
            return (
                f'        <bendpoint id="{self.id}"\n'
                f'                   serial="{self.serial}">\n'
                f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                f'        </bendpoint>\n'
            )

    class Annot:
        def __init__(self, id_):
            self.id = id_
            self.text = None
            self.posattr = BasicNode.Posattr()
            self.fillattr = BasicNode.Fillattr()
            self.lineattr = BasicNode.Lineattr()
            self.textattr = BasicNode.Textattr()

        def __str__(self):
            if self.text:
                return (
                    f'        <annot id="{self.id}">\n'
                    f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                    f'          <text tool="CPN Tools"\n'
                    f'                version="4.0.0">{self.text}</text>\n'
                    f'        </annot>\n'
                )
            else:
                return (
                    f'        <annot id="{self.id}">\n'
                    f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                    f'          <text tool="CPN Tools"\n'
                    f'                version="4.0.0"/>\n'
                    f'        </annot>\n'
                )

    def __init__(self, transend=None, placeend=None, orientation=None):
        super().__init__()
        self._rand = random.Random()
        self._id_counter = self._rand.randint(3000, 3999)
        self.orientation = orientation
        self.order = 1
        self.transend = transend
        self.placeend = placeend
        self.arrowattr = ArcNode.Arrowattr()
        self.bendpoint = None
        self.annot = ArcNode.Annot(self._generate_id())
        self.set_posattr(0.0, 0.0)  # 默认位置

    def _generate_id(self):
        self._id_counter += 1
        return f"ID{self._id_counter}"

    def set_annot(self, text, x, y,id_generator):
        self.annot.id = id_generator()
        self.annot.text = text
        self.annot.posattr.x = x
        self.annot.posattr.y = y

    def set_bendpoint(self, serial, x, y):
        self.bendpoint = ArcNode.Bendpoint(self._generate_id())
        self.bendpoint.serial = serial
        self.bendpoint.posattr.x = x
        self.bendpoint.posattr.y = y

    def set_arrowattr(self, headsize, currentcyckle):
        self.arrowattr.headsize = headsize
        self.arrowattr.currentcyckle = currentcyckle

    def __str__(self):
        arc = (
            f'      <arc id="{self.ID}"\n'
            f'           orientation="{self.orientation}"\n'
            f'           order="{self.order}">\n'
            f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
            f'{self.arrowattr}'
            f'        <transend idref="{self.transend}"/>\n'
            f'        <placeend idref="{self.placeend}"/>\n'
        )
        if self.bendpoint and self.bendpoint.posattr.x is not None:
            arc += str(self.bendpoint)
        arc += str(self.annot)
        arc += "      </arc>\n"
        return arc
