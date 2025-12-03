from typing import Optional, List


class ColorNode:
    class Product:
        def __init__(self, sum_: int = 1, clo: Optional[List[str]] = None, tag: str = "product"):
            """
            兼容性较强的构造函数：
            - 旧签名：Product(sum_, clo)
            - 新签名：Product(sum_, clo, tag='list')
            - 若首参数被误传为 list（即 Product(['REAL'])），会自动认为这是 clo 并修正 sum_。
            """
            # 如果用户把 clo 误放在第一个位置（例如 Product(['REAL'])），则修正参数
            if isinstance(sum_, list):
                clo = sum_
                sum_ = len(clo)

            # 如果 clo 为字符串（单个类型），把它包装成 list
            if isinstance(clo, str):
                clo = [clo]

            self.tag = tag or "product"
            self.sum = int(sum_) if sum_ is not None else (len(clo) if clo else 1)
            self.clo = clo if clo is not None else [""]

        def __str__(self):
            # 使用 self.tag 作为外层标签名（动态）
            pro = f"          <{self.tag}>\n"
            for i in range(self.sum):
                val = self.clo[i] if i < len(self.clo) else ""
                pro += f"            <id>{val}</id>\n"
            pro += f"          </{self.tag}>\n"
            return pro

    def __init__(self):
        self.id: Optional[str] = None
        self.name: Optional[str] = None
        self.layout: Optional[str] = None
        self.product = self.Product()

    def getId(self) -> Optional[str]:
        return self.id

    def setId(self, id_: str):
        self.id = id_

    def getName(self) -> Optional[str]:
        return self.name

    def setName(self, name: str):
        self.name = name
        self.setProduct()
        self.setLayout()

    def getLayout(self) -> Optional[str]:
        return self.layout

    def setLayout(self):
        self.layout = f"colset {self.name}=product {self.name.replace('x', '*')};"

    def setProduct(self):
        sum_ = self.getProSum()
        clo = self.name.split("x") if self.name else [""]
        self.product = self.Product(sum_, clo)

    def getProSum(self) -> int:
        if not self.name:
            return 1
        return self.name.count('x') + 1

    def __str__(self):
        return (
            f'        <color id="{self.id}">\n'
            f'          <id>{self.name}</id>\n'
            f'{self.product}'
            f'          <layout>{self.layout}</layout>\n'
            f'        </color>\n'
        )
