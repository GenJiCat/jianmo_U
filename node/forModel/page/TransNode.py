import random

from node.forModel.page.BasicNode import BasicNode

class TransNode(BasicNode):
    def __init__(self):
        super().__init__()
        self.explicit = False
        self.box = self.Box()
        self.subst = self.Subst()
        self.binding = self.Binding()
        self.cond = self.Cond()
        self.time = self.Time()
        self.code = self.Code()
        self.priority = self.Priority()
        self.channel = self.Channel()
        self.r = random.Random()
        # 不再使用随机小范围的 self.Id 作为局部计数器（会导致重复）
        # 使用本地序号并基于父级 self.ID 生成唯一子 ID
        self._local_seq = 1

    def _make_local_id(self) -> str:
        """
        生成 Trans 子元素（cond/time/code/priority/channel/subst.sid 等）的唯一 ID。
        主要策略：如果 self.ID 存在且形如 'ID<number>'，以该 number 为 base，生成
        ID{base*100 + seq} 的 ID；否则使用较大的随机数回退。
        这样生成的 ID 与全局由 CreatModel 生成的 self.ID 空间冲突概率非常低。
        """
        seq = self._local_seq
        self._local_seq += 1
        # try to derive numeric base from self.ID (set by CreatModel when node id assigned)
        base_num = None
        try:
            if hasattr(self, 'ID') and isinstance(self.ID, str) and self.ID.startswith('ID'):
                base_num = int(self.ID[2:])
        except Exception:
            base_num = None

        if base_num is not None:
            new_num = base_num * 100 + seq
        else:
            # fallback: large random number to avoid collision
            new_num = self.r.randint(10_000_000, 99_999_999)

        return f"ID{new_num}"

    class Box:
        def __init__(self):
            self.w = 70.0
            self.h = 40.0

        def __str__(self):
            return f'        <box w="{self.w}"\n             h="{self.h}"/>\n'

    class Subst(BasicNode):
        def __init__(self):
            super().__init__()
            self.subpage = None
            self.portsock = None
            self.sid = None
            self.sname = None

        def __str__(self):
            if not self.subpage:
                return ''
            return (
                f'        <subst subpage="{self.subpage}"\n'
                f'               portsock="{self.portsock}">\n'
                f'          <subpageinfo id="{self.sid}"\n'
                f'                       name="{self.sname}">\n'
                f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                f'          </subpageinfo>\n'
                f'        </subst>\n'
            )

    class Binding:
        def __init__(self):
            self.x = 7.0
            self.y = -3.0

        def __str__(self):
            return f'        <binding x="{self.x}"\n                 y="{self.y}"/>\n'

    class Cond(BasicNode):
        def __init__(self):
            super().__init__()
            self.id = None
            self.text = None

        def __str__(self):
            if not self.id:
                return ''
            body = (
                f"          <text tool=\"CPN Tools\"\n"
                f"                version=\"4.0.0\">{self.text}</text>\n"
                if self.text else
                "          <text tool=\"CPN Tools\"\n"
                "                version=\"4.0.0\"/>\n"
            )
            return (
                f'        <cond id="{self.id}">\n'
                f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
                f'{body}'
                f'        </cond>\n'
            )

    class Time(Cond):
        def __str__(self):
            if not self.id:
                return ''
            return super().__str__().replace('<cond', '<time').replace('</cond>', '</time>')

    class Code(Cond):
        def __str__(self):
            if not self.id:
                return ''
            return super().__str__().replace('<cond', '<code').replace('</cond>', '</code>')

    class Priority(Cond):
        def __str__(self):
            if not self.id:
                return ''
            return super().__str__().replace('<cond', '<priority').replace('</cond>', '</priority>')

    class Channel(Cond):
        def __str__(self):
            if not self.id:
                return ''
            return super().__str__().replace('<cond', '<channel').replace('</cond>', '</channel>')

    def set_subst(self, subpage, portsock, name):
        # 如果已经设置了 sid，则不要覆盖（避免重复）
        if not self.subst.sid:
            self.subst.sid = self._make_local_id()
        self.subst.subpage = subpage
        self.subst.portsock = portsock
        self.subst.sname = name
        # posattr.x/y 可能是 None，如果是则保护性赋默认
        try:
            self.subst.posattr.x = int(self.posattr.x) + 50
        except Exception:
            self.subst.posattr.x = 0
        try:
            self.subst.posattr.y = int(self.posattr.y) - 40
        except Exception:
            self.subst.posattr.y = 0

    def set_cond(self, text=None):
        # 只在没有 id 时生成 id（避免覆盖已有 id）
        if not getattr(self.cond, 'id', None):
            self.cond.id = self._make_local_id()
        self.cond.text = text
        try:
            self.cond.posattr.x = int(self.posattr.x) - 40
        except Exception:
            self.cond.posattr.x = 0
        try:
            self.cond.posattr.y = int(self.posattr.y) + 30
        except Exception:
            self.cond.posattr.y = 0

    def set_time(self, text=None):
        if not getattr(self.time, 'id', None):
            self.time.id = self._make_local_id()
        self.time.text = text
        try:
            self.time.posattr.x = int(self.posattr.x) + 50
        except Exception:
            self.time.posattr.x = 0
        try:
            self.time.posattr.y = int(self.posattr.y) + 30
        except Exception:
            self.time.posattr.y = 0

    def set_code(self, text=None):
        if not getattr(self.code, 'id', None):
            self.code.id = self._make_local_id()
        self.code.text = text
        try:
            self.code.posattr.x = int(self.posattr.x) + 70
        except Exception:
            self.code.posattr.x = 0
        try:
            self.code.posattr.y = int(self.posattr.y) - 60
        except Exception:
            self.code.posattr.y = 0

    def set_priority(self, text=None):
        if not getattr(self.priority, 'id', None):
            self.priority.id = self._make_local_id()
        self.priority.text = text
        try:
            self.priority.posattr.x = int(self.posattr.x) - 70
        except Exception:
            self.priority.posattr.x = 0
        try:
            self.priority.posattr.y = int(self.posattr.y) - 30
        except Exception:
            self.priority.posattr.y = 0

    def set_channel(self, text=None):
        if not getattr(self.channel, 'id', None):
            self.channel.id = self._make_local_id()
        self.channel.text = text
        try:
            self.channel.posattr.x = int(self.posattr.x)
        except Exception:
            self.channel.posattr.x = 0
        try:
            self.channel.posattr.y = int(self.posattr.y)
        except Exception:
            self.channel.posattr.y = 0

    def set_other_labels(self):
        # 调用 set_* 方法，但每个 set_* 都检查是否已存在 id，避免重复覆盖
        # 仅在需要时生成 id
        self.set_cond(self.cond.text)
        self.set_time(self.time.text)
        self.set_code(self.code.text)
        self.set_priority(self.priority.text)
        self.set_channel(self.channel.text)

    def __str__(self):
        base = (
            f'      <trans id="{self.ID}"\n'
            f'             explicit="{str(self.explicit).lower()}">\n'
            f'{self.posattr}{self.fillattr}{self.lineattr}{self.textattr}'
            f'        <text>{self.text}</text>\n'
            f'{self.box}'
        )
        parts = []
        if self.subst.subpage:
            parts.append(str(self.subst))
        parts.append(str(self.binding))
        # only include non-empty optional parts
        for part in (self.cond, self.time, self.code, self.priority, self.channel):
            s = str(part)
            if s:
                parts.append(s)
        return base + ''.join(parts) + '      </trans>\n'
