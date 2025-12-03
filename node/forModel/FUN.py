class Fun:
    def __init__(self, name=None, type_=None):
        self.name = name  # 函数名称，如 fun1
        self.type = type_  # 函数类型，如 if, ifelse, for, while, assignment
        self.fun_text = []  # 存储语句内容

    def set_type(self, type_):
        self.type = type_

    def get_type(self):
        return self.type

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def add_text(self, text):
        self.fun_text.append(text)

    def get_text(self):
        return self.fun_text

    def __str__(self):
        return f"Fun(name='{self.name}', type='{self.type}', funText={self.fun_text})"
