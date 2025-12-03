import random
import re
# from node. import FusionNode

from node.forModel.globbox.MINode import MlNode
from node.forModel.globbox.VarMoNode import VarMoNode
from node.forModel.insbin.BinderNode import BinderNode
from node.forModel.insbin.InstanceNode import InstanceNode
from node.forModel.page.ArcNode import ArcNode
from node.forModel.page.PageNode import PageNode
from node.forModel.page.PlaceNode import PlaceNode
from node.forModel.page.TransNode import TransNode
from node.forModel.globbox.ColorNode import ColorNode



class CreatModel:
    def __init__(self):
        self.cpntxt = []
        self.r = random.Random()
        # keep deterministic small start, but we'll re-sync with template in setCpntxt()
        self.Id = 1
        self.pageNo = 1
        self.placeNo = 1
        self.transNo = 1
        self.fusionNo = 1
        self.zorder = 0
        self.trans_nodes = {}
        # set to track all used IDs (strings like 'ID12345')
        self._existing_ids = set()

    def _init_id_from_cpntxt(self):
        """
        Scan current cpntxt for any existing ID\d+ occurrences and
        initialize self._existing_ids and self.Id accordingly so new IDs won't collide.
        """
        maxnum = 0
        pattern = re.compile(r'ID(\d+)')
        for line in self.cpntxt:
            for m in pattern.finditer(line):
                try:
                    n = int(m.group(1))
                    self._existing_ids.add(f"ID{n}")
                    if n > maxnum:
                        maxnum = n
                except ValueError:
                    continue
        # set next id to be greater than any found
        self.Id = maxnum + 1 if maxnum >= 1 else max(self.Id, 1)

    def get_new_id(self) -> str:
        """
        Return a new unique ID string like 'ID123', guaranteed not to be in _existing_ids.
        """
        # loop until we find an unused id number
        while True:
            candidate = f"ID{self.Id}"
            self.Id += 1
            if candidate not in self._existing_ids:
                self._existing_ids.add(candidate)
                return candidate

    def getCpntxt(self):
        return self.cpntxt

    def setCpntxt(self, cpntxt):
        self.cpntxt = cpntxt
        # reinitialize existing ids based on the template we just loaded
        self._init_id_from_cpntxt()

    def addPage(self, news, subtrans, instance):
        page = PageNode()
        page.set_name(f"New Page{self.pageNo}")
        self.pageNo += 1
        new_id = self.get_new_id()
        page.set_id(new_id)
        if instance:
            self.addInstance(page.get_id(), subtrans)
        for item in news:
            page.set_pageshow(item)
        idx = next((i for i in range(len(self.cpntxt)-1)
                    if ("</globbox>" in self.cpntxt[i] or "</page>" in self.cpntxt[i])
                    and ("<instances>" in self.cpntxt[i+1] or "<fusion" in self.cpntxt[i+1])),
                   len(self.cpntxt)-1)
        self.cpntxt.insert(idx+1, page.__str__())
        return page.get_id()

    def addInstance(self, page_id, subtrans):
        inst = InstanceNode()
        inst_id = self.get_new_id()
        inst.set_id(inst_id)
        inst.set_page(page_id)
        if subtrans:
            inst.set_trans(subtrans)
        for iid in inst.get_instance_id():
            self.addBinders(iid)
        idx = next((i for i in range(len(self.cpntxt)-1)
                    if "<instances>" in self.cpntxt[i] and "</instances>" in self.cpntxt[i+1]),
                   len(self.cpntxt)-1)
        self.cpntxt.insert(idx+1, inst.__str__())

    def addBinders(self, instance_id):
        binder = BinderNode(instance_id)
        idx = next((i for i in range(len(self.cpntxt)-1)
                    if ("<sheets>" in self.cpntxt[i] or "</cpnsheet>" in self.cpntxt[i])
                    and "</sheets>" in self.cpntxt[i+1]),
                   len(self.cpntxt)-1)
        self.cpntxt.insert(idx+1, binder.__str__())
        j = next((j for j in range(len(self.cpntxt)-2)
                  if ("<zorder>" in self.cpntxt[j] or "<position value=" in self.cpntxt[j])
                  and "</zorder>" in self.cpntxt[j+1] and "</cpnbinder>" in self.cpntxt[j+2]),
                 len(self.cpntxt)-2)
        pos = f"          <position value=\"{self.zorder}\"/>\n"
        self.zorder += 1
        self.cpntxt.insert(j+1, pos)

    def addPlace(self, x, y, type_, black=False):
        node = PlaceNode()
        node_id = self.get_new_id()
        node.set_id(node_id)
        node.set_text(f"P{self.placeNo}")
        self.placeNo += 1
        node.set_posattr(x, y)
        type_id = self.get_new_id()
        node.set_type(type_, type_id)
        if black:
            node.set_fillattr_black()
        initmark_id =self.get_new_id()
        type_id = self.get_new_id()
        node.set_other_labels(type_id, initmark_id)
        return [node.__str__(), node.get_id(), node.get_px(), node.get_py()]

    def addPlace_init(self, x, y, type_, initmark, black=False):
        node = PlaceNode()
        node_id = self.get_new_id()
        node.set_id(node_id)
        node.set_text(f"P{self.placeNo}")
        self.placeNo += 1
        node.set_posattr(x, y)
        type_id = self.get_new_id()
        node.set_type(type_,type_id)
        initmark_id=self.get_new_id()
        print(initmark)
        # print(initmark_id)
        node.set_initmark(initmark,initmark_id)
        if black:
            node.set_fillattr_black()
        initmark_id = self.get_new_id()
        type_id = self.get_new_id()
        node.set_other_labels(type_id, initmark_id)
        return [node.__str__(), node.get_id(), node.get_px(), node.get_py()]

    def addPortPlace(self, x, y, type_, porttype):
        node = PlaceNode()
        node_id = self.get_new_id()
        node.set_id(node_id)
        node.set_text(f"P{self.placeNo}")
        self.placeNo += 1
        node.set_posattr(x, y)
        type_id = self.get_new_id()
        node.set_type(type_, type_id)
        print(porttype)
        node.set_port(porttype)
        initmark_id = self.get_new_id()
        type_id = self.get_new_id()
        node.set_other_labels(type_id, initmark_id)
        return [node.__str__(), node.get_id(), node.get_px(), node.get_py()]

    def addFusionPlace(self, x, y, type_, fusname):
        node = PlaceNode()
        node_id = self.get_new_id()
        node.set_id(node_id)
        node.set_text(f"P{self.placeNo}")
        self.placeNo += 1
        node.set_posattr(x, y)
        type_id = self.get_new_id()
        node.set_type(type_, type_id)
        node.set_fusioninfo(fusname)
        initmark_id = self.get_new_id()
        type_id = self.get_new_id()
        node.set_other_labels(type_id, initmark_id)
        return [node.__str__(), node.get_id(), node.get_px(), node.get_py()]



    def addTrance(self, x, y, black=False,):
        node = TransNode()
        node_id = self.get_new_id()
        node.set_id(node_id)
        node.set_text(f"T{self.transNo}")
        self.transNo += 1
        node.set_posattr(x, y)
        if black:
            node.set_fillattr_black()
        self.trans_nodes[node.get_id()] = node
        return [node.__str__(), node.get_id(), node.get_px(), node.get_py()]

    def addTrance_file(self, x, y,code_text, black=False ):
        node = TransNode()
        node_id = self.get_new_id()
        node.set_id(node_id)
        node.set_text(f"T{self.transNo}")
        self.transNo += 1
        node.set_posattr(x, y)
        node.set_code(code_text)
        if black:
            node.set_fillattr_black()
        self.trans_nodes[node.get_id()] = node
        return [node.__str__(), node.get_id(), node.get_px(), node.get_py()]
    def getTransNode(self, tid):
        return self.trans_nodes.get(tid)

    def addSubTrance(self, x, y, subpage, portsock, sname):
        node = TransNode()
        node_id = self.get_new_id()
        node.set_id(node_id)
        node.set_text(f"T{self.transNo}")
        self.transNo += 1
        node.set_posattr(x, y)
        node.set_subst(subpage, portsock, sname)
        node.set_other_labels()
        return [node.__str__(), node.get_id(), node.get_px(), node.get_py()]

    def addArc(self, transend, placeend, orientation, text, ax, ay):
        arc = ArcNode(transend, placeend, orientation)
        arc_id = self.get_new_id()
        arc.set_id(arc_id)
        arc.set_annot(text, ax, ay,self.get_new_id)
        return arc.__str__()

    # --- 放到 class CreatModel 内 ---

    def _find_globbox_end_index(self) -> int:
        """
        返回应当插入全局定义的位置（即在 </globbox> 之前的行索引）。
        如果没有找到 </globbox>，退而求其次找到 <instances> 之前；否则返回文件末尾。
        """
        for i, line in enumerate(self.cpntxt):
            if '</globbox>' in line:
                return i
        # 如果没有 </globbox>，尝试找到 <instances>
        for i, line in enumerate(self.cpntxt):
            if '<instances>' in line:
                return i
        # 最后回退到文件末尾
        return len(self.cpntxt)

    def addGlobRef(self, name: str, ml: str) -> str:
        """
        在 <globbox> 区域插入 <globref> 节点。
        name: 标识符 (eg "m2")
        ml:   ml 内容 (eg "empty : ELEM2 ms")
        返回自动分配的 XML id 字符串 (eg "ID12345")
        """
        # 简单 XML 转义，防止 ml 中包含 <>& 等字符导致结构错误
        esc_ml = ml.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        node_id = self.get_new_id()
        layout = f"globref {name} = {ml};"
        globref_str = (
            f'        <globref id="{node_id}">\n'
            f'          <id>{name}</id>\n'
            f'          <ml>{esc_ml}</ml>\n'
            f'          <layout>{layout}</layout>\n'
            f'        </globref>\n'
        )
        idx = self._find_globbox_end_index()
        # 插入到 </globbox> 之前
        self.cpntxt.insert(idx, globref_str)
        return node_id

    def addColor(self, name):
        """
        稳健版 addColor。支持两种调用方式：
          1) addColor('LR')            -> 使用 ColorNode（保持原行为）
          2) addColor(('list','LR','REAL')) -> 生成 <list> 标签并 layout 为 colset LR = list REAL;
        插入位置保证在 </globbox> 之前。
        """
        idx = self._find_globbox_end_index()

        if isinstance(name, (list, tuple)):
            # 处理 ('list','LR','REAL') 形式
            if len(name) >= 3 and name[0] == 'list':
                typ = name[0]
                colname = name[1]
                inner = name[2]
                layout = f"colset {colname} = {typ} {inner};"
                color_id = self.get_new_id()
                color_str = (
                    f'        <color id="{color_id}">\n'
                    f'          <id>{colname}</id>\n'
                    f'          <{typ}>\n'
                    f'            <id>{inner}</id>\n'
                    f'          </{typ}>\n'
                    f'          <layout>{layout}</layout>\n'
                    f'        </color>\n'
                )
                self.cpntxt.insert(idx, color_str)
                return color_id
            else:
                # 其他复杂情况暂回退到字符串化处理
                colname = str(name)
        else:
            colname = name

        # 保持向后兼容：用你的 ColorNode（如果存在）构造字符串
        try:
            node = ColorNode()
            node_id = self.get_new_id()
            node.setId(node_id)
            node.setName(colname)
            self.cpntxt.insert(idx, node.__str__())
            return node_id
        except Exception:
            # 若没有 ColorNode 或发生异常，退回简单文本插入
            color_id = self.get_new_id()
            layout = f"colset {colname} = {colname};"
            color_str = (
                f'        <color id="{color_id}">\n'
                f'          <id>{colname}</id>\n'
                f'          <layout>{layout}</layout>\n'
                f'        </color>\n'
            )
            self.cpntxt.insert(idx, color_str)
            return color_id

    def addVar(self, colset_name: str, var_name: str) -> str:
        """
        在 <globbox> 区域插入一个完整的 <var> 节点：
          <var id="...">
            <type><id>COLSET</id></type>
            <id>VARNAME</id>
            <layout>var VARNAME:COLSET;</layout>
          </var>
        参数:
          colset_name: 颜色集名，例如 "LR"
          var_name:     变量名，例如 "x"
        返回: 新生成的 XML id 字符串 (例如 "ID12345")
        """
        idx = self._find_globbox_end_index()
        var_id = self.get_new_id()

        # 构造 layout 文本（不用转义 ':' 等，因为这是 SML 形式）
        layout_text = f"var {var_name}:{colset_name};"

        var_str = (
            f'        <var id="{var_id}">\n'
            f'          <type>\n'
            f'            <id>{colset_name}</id>\n'
            f'          </type>\n'
            f'          <id>{var_name}</id>\n'
            f'          <layout>{layout_text}</layout>\n'
            f'        </var>\n'
        )

        # 将 var 插入到 </globbox> 之前（或合适的回退位置）
        self.cpntxt.insert(idx, var_str)
        return var_id

    def addMl(self, funtext: str) -> str:
        """
        将一段 ML (SML) 文本作为 <ml> 节点插入到 <globbox> 区域内（在 </globbox> 之前）。
        funtext: 要插入的函数文本（原样，MlNode.__str__ 应当负责正确包裹 <ml>/<layout> 等）
        返回新生成的 XML id（例如 "ID12345"）
        """
        node = MlNode()
        node_id = self.get_new_id()
        node.setId(node_id)
        node.setText(funtext)

        # 找到插入位置（确保在 globbox 区域）
        idx = self._find_globbox_end_index()

        # 把 node 的字符串插入到该位置（在 </globbox> 之前）
        # 注意：使用 node.__str__() 保留 MlNode 内部格式（包括 <ml> 与 <layout>）
        self.cpntxt.insert(idx, node.__str__())
        return node_id

    # def addFusion(self, places):
    #     node = FusionNode()
    #     node.setId(f"ID{self.fusionNo}")
    #     self.Id += 1
    #     node.setName(f"Fusion {self.fusionNo}")
    #     self.fusionNo += 1
    #     node.setIdref(places)
    #     i = next((i for i in range(len(self.cpntxt)-1)
    #               if "</page>" in self.cpntxt[i] and "<instances>" in self.cpntxt[i+1]),
    #              len(self.cpntxt)-1)
    #     self.cpntxt.insert(i+1, node.toString())

    # 占位方法


