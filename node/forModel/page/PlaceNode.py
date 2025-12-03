import random
from node.forModel.page.BasicNode import BasicNode


class PlaceNode(BasicNode):
    def __init__(self):
        super().__init__()
        self._r = random.Random()
        self._id_counter = self._r.randint(1000, 1999)

        self.ellipse = self.Ellipse()
        self.token = self.Token()
        self.marking = self.Marking()
        self.type = self.Type()
        self.initmark = self.Initmark()
        self.port = None
        self.fusioninfo = None

    def _gen_id(self):
        self._id_counter += 1
        return f"ID{self._id_counter}"

    # ===== Ellipse =====
    class Ellipse:
        def __init__(self):
            self.w = 60.0
            self.h = 40.0

        def __str__(self):
            return f'        <ellipse w="{self.w}"\n                 h="{self.h}"/>\n'

    def set_ellipse(self, w, h):
        self.ellipse.w = w
        self.ellipse.h = h

    # ===== Token =====
    class Token:
        def __init__(self):
            self.x = -10.0
            self.y = 0.0

        def __str__(self):
            return f'        <token x="{self.x}"\n               y="{self.y}"/>\n'

    def set_token(self, x, y):
        self.token.x = x
        self.token.y = y

    # ===== Marking =====
    class Marking:
        def __init__(self):
            self.x = -25.0
            self.y = 35.0
            self.hidden = False
            self.snap = ('snap_id="0"\n'
                         '                anchor.horizontal="0"\n'
                         '                anchor.vertical="0"')

        def __str__(self):
            return (f'        <marking x="{self.x}"\n'
                    f'                 y="{self.y}"\n'
                    f'                 hidden="{str(self.hidden).lower()}">\n'
                    f'          <snap {self.snap}/>\n'
                    f'        </marking>\n')

    def set_marking(self, x, y, snap=None):
        self.marking.x = x
        self.marking.y = y
        if snap:
            self.marking.snap = snap

    # ===== Type =====
    class Type:
        def __init__(self):
            self.id = None
            self.posattr = BasicNode.Posattr()
            self.fillattr = BasicNode.Fillattr()
            self.lineattr = BasicNode.Lineattr()
            self.textattr = BasicNode.Textattr()
            self.text = None
            self.fillattr.pattern = "Solid"
            self.lineattr.thick = 0

        def __str__(self):
            text_part = f"{self.text}" if self.text else ""
            return (f'        <type id="{self.id}">\n'
                    f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                    f'          <text tool="CPN Tools"\n'
                    f'                version="4.0.0">{text_part}</text>\n'
                    f'        </type>\n')

    def set_type(self, text=None, id_value=None):
        """
        如果传入 id_value，使用该 ID；否则回退到旧的内部生成器 _gen_id().
        同时保证 posattr.x/y 有合理默认值，避免 None 写入 XML。
        """
        if id_value is None:
            # 兼容旧逻辑：内部生成 id
            try:
                self.type.id = self._gen_id()
            except Exception:
                # 最后手段，使用字符串占位
                self.type.id = "ID0"
        else:
            # 使用传入的全局唯一 ID（确保为字符串）
            self.type.id = str(id_value)

        # 防御性设置位置（避免 None）
        try:
            self.type.posattr.x = int(self.posattr.x) + 45
        except Exception:
            self.type.posattr.x = 0
        try:
            self.type.posattr.y = int(self.posattr.y) - 24
        except Exception:
            self.type.posattr.y = 0

        if text:
            self.type.text = text

    # ===== Initmark =====
    class Initmark:
        def __init__(self):
            self.id = None
            self.posattr = BasicNode.Posattr()
            self.fillattr = BasicNode.Fillattr()
            self.lineattr = BasicNode.Lineattr()
            self.textattr = BasicNode.Textattr()
            self.text = None
            self.fillattr.pattern = "Solid"
            self.lineattr.thick = 0

        def __str__(self):
            text_part = f"{self.text}" if self.text else ""
            return (f'        <initmark id="{self.id}">\n'
                    f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                    f'          <text tool="CPN Tools"\n'
                    f'                version="4.0.0">{text_part}</text>\n'
                    f'        </initmark>\n')

    def set_initmark(self, text=None, id_value=None):
        """
        如果传入 id_value，则使用该 ID（由 CreatModel.get_new_id() 提供）。
        否则回退到原有的内部生成器（保持向后兼容）。
        """
        # print(id_value)
        if id_value is None:
            # 旧逻辑：内部生成（保留以兼容其他调用）
            self.initmark.id = self._gen_id()
        else:
            # 使用外部提供的全局唯一 ID
            self.initmark.id = str(id_value)

        # 防御性设置位置，避免 None
        try:
            self.initmark.posattr.x = int(self.posattr.x) + 56
            self.initmark.posattr.y = int(self.posattr.y) + 50
        except Exception:
            # 若 posattr 未初始化或不是数字，则使用安全默认
            self.initmark.posattr.x = 0
            self.initmark.posattr.y = 0

        if text:
            self.initmark.text = text

    # ===== Port =====
    class Port:
        def __init__(self):
            self.id = None
            self.type = None
            self.posattr = BasicNode.Posattr()
            self.fillattr = BasicNode.Fillattr()
            self.lineattr = BasicNode.Lineattr()
            self.textattr = BasicNode.Textattr()
            self.fillattr.pattern = "Solid"
            self.lineattr.thick = 0

        def __str__(self):
            return (f'        <port id="{self.id}"\n'
                    f'              type="{self.type}">\n'
                    f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                    f'        </port>\n')

    def set_port(self, port_type):
        self.port = self.Port()
        self.port.id = self._gen_id()
        self.port.type = port_type
        self.port.posattr.x = self.posattr.x - 35
        self.port.posattr.y = self.posattr.y - 25

    # ===== Fusioninfo =====
    class Fusioninfo:
        def __init__(self):
            self.id = None
            self.name = None
            self.posattr = BasicNode.Posattr()
            self.fillattr = BasicNode.Fillattr()
            self.lineattr = BasicNode.Lineattr()
            self.textattr = BasicNode.Textattr()
            self.lineattr.thick = 0

        def __str__(self):
            return (f'        <fusioninfo id="{self.id}"\n'
                    f'                    name="{self.name}">\n'
                    f'          <posattr x="{self.posattr.x}"\n'
                    f'                   y="{self.posattr.y}"/>\n'
                    f'          <fillattr colour="{self.fillattr.colour}"\n'
                    f'                    pattern="{self.fillattr.pattern}"\n'
                    f'                    filled="{str(self.fillattr.filled).lower()}"/>\n'
                    f'          <lineattr colour="{self.lineattr.colour}"\n'
                    f'                    thick="{self.lineattr.thick}"\n'
                    f'                    type="{self.lineattr.type}"/>\n'
                    f'          <textattr colour="{self.textattr.colour}"\n'
                    f'                    bold="{str(self.textattr.bold).lower()}"/>\n'
                    f'        </fusioninfo>\n')

    def set_fusioninfo(self, name):
        self.fusioninfo = self.Fusioninfo()
        self.fusioninfo.id = self._gen_id()
        self.fusioninfo.name = name
        self.fusioninfo.posattr.x = self.posattr.x - 38
        self.fusioninfo.posattr.y = self.posattr.y - 23

    # ===== Label设置汇总 =====
    def set_other_labels(self, id_value=None):
        """
        其它标签的统一入口。若给出 id_value，传递给 set_initmark，
        保证 initmark 使用全局 id 分配。
        """
        self.set_type()
        # 将 id_value 传递下去（可能为 None）
        self.set_initmark(None, id_value)

    def __str__(self):
        base = (
            f'      <place id="{self.ID}">\n'
            f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
            f'        <text>{self.text}</text>\n'
            f'{self.ellipse}{self.token}{self.marking}{self.type}{self.initmark}'
        )
        if self.port:
            base += str(self.port)
        if self.fusioninfo:
            base += str(self.fusioninfo)
        base += '      </place>\n'
        return base
