class PageNode:
    def __init__(self):
        self.id = None
        self.name = "New Page"
        self.place_node = None
        self.trans_node = None
        self.arc_node = None
        self.pageshow = ""
        self.sub_trans = []  # 存储本页面的替代变迁 id（strings）

    def get_id(self):
        return self.id

    def set_id(self, id_):
        self.id = id_
        self.pageshow = (
            f'    <page id="{self.id}">\n'
            f'      <pageattr name="{self.name}"/>\n'
        )

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_pageshow(self):
        return self.pageshow

    def set_pageshow(self, new_content):
        self.pageshow += new_content

    def __str__(self):
        return self.pageshow + "      <constraints/>\n" + "    </page>\n"
class PageNode:
    def __init__(self):
        self.id = None
        self.name = "New Page"
        self.place_node = None
        self.trans_node = None
        self.arc_node = None
        self.pageshow = ""
        self.sub_trans = []  # 存储本页面的替代变迁 id（strings）

    def get_id(self):
        return self.id

    def set_id(self, id_):
        self.id = id_
        self.pageshow = (
            f'    <page id="{self.id}">\n'
            f'      <pageattr name="{self.name}"/>\n'
        )

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_pageshow(self):
        return self.pageshow

    def set_pageshow(self, new_content):
        self.pageshow += new_content

    def __str__(self):
        return self.pageshow + "      <constraints/>\n" + "    </page>\n"
