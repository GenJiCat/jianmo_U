class BasicNode:
    class Posattr:
        def __init__(self):
            self.x = None
            self.y = None

        def __str__(self):
            return (
                f'        <posattr x="{self.x}"\n'
                f'                 y="{self.y}"/>\n'
            )

    class Fillattr:
        def __init__(self):
            self.colour = "White"
            self.pattern = ""
            self.filled = False

        def __str__(self):
            return (
                f'        <fillattr colour="{self.colour}"\n'
                f'                  pattern="{self.pattern}"\n'
                f'                  filled="{str(self.filled).lower()}"/>\n'
            )

    class Lineattr:
        def __init__(self):
            self.colour = "Black"
            self.thick = 1
            self.type = "solid"

        def __str__(self):
            return (
                f'        <lineattr colour="{self.colour}"\n'
                f'                  thick="{self.thick}"\n'
                f'                  type="{self.type}"/>\n'
            )

    class Textattr:
        def __init__(self):
            self.colour = "Black"
            self.bold = False

        def __str__(self):
            return (
                f'        <textattr colour="{self.colour}"\n'
                f'                  bold="{str(self.bold).lower()}"/>\n'
            )

    def __init__(self, ID=None, posattr=None, fillattr=None, lineattr=None, textattr=None, text=None):
        self.ID = ID
        self.posattr = posattr if posattr else BasicNode.Posattr()
        self.fillattr = fillattr if fillattr else BasicNode.Fillattr()
        self.lineattr = lineattr if lineattr else BasicNode.Lineattr()
        self.textattr = textattr if textattr else BasicNode.Textattr()
        self.text = text

    def set_posattr(self, x, y):
        self.posattr.x = x
        self.posattr.y = y

    def get_px(self):
        return str(self.posattr.x)

    def get_py(self):
        return str(self.posattr.y)

    def get_fillattr(self):
        return self.fillattr

    def set_fillattr_black(self):
        self.fillattr.colour = "Black"
        self.textattr.colour = "White"

    def get_lineattr(self):
        return self.lineattr

    def set_lineattr(self, colour, thick, type_):
        self.lineattr.colour = colour
        self.lineattr.thick = thick
        self.lineattr.type = type_

    def get_textattr(self):
        return self.textattr

    def set_textattr(self, colour, bold):
        self.textattr.colour = colour
        self.textattr.bold = bold

    def get_id(self):
        return self.ID

    def set_id(self, ID):
        self.ID = ID

    def get_text(self):
        return self.text

    def set_text(self, text):
        self.text = text

    def __str__(self):
        return (
            f"BasicNode{{"
            f"ID='{self.ID}', "
            f"posattr={self.posattr}, "
            f"fillattr={self.fillattr}, "
            f"lineattr={self.lineattr}, "
            f"textattr={self.textattr}, "
            f"text='{self.text}'"
            f"}}"
        )
