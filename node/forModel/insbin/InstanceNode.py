import random

class InstanceNode:
    def __init__(self, id=None, page=None, trans=None):
        self.id = id
        self.page = page
        self.trans = trans
        self.instanceID = []
        self._id_counter = random.randint(4000, 4999)

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_page(self):
        return self.page

    def set_page(self, page):
        self.page = page

    def get_trans(self):
        return self.trans

    def set_trans(self, subtrans):
        self.instanceID.append(self.id)
        self.trans = ""

        if len(subtrans) == 1:
            for i in range(len(subtrans[0])):
                trans_id = f"ID{self._id_counter}"
                self._id_counter += 1
                self.instanceID.append(trans_id)
                self.trans += (
                    f"        <instance id=\"{trans_id}\"\n"
                    f"                  trans=\"{subtrans[0][i]}\"/>\n"
                )
        else:
            for i in reversed(range(len(subtrans))):
                trans_id = f"ID{self._id_counter}"
                self._id_counter += 1
                self.instanceID.append(trans_id)
                if i == 0:
                    self.trans += (
                        f"        <instance id=\"{trans_id}\"\n"
                        f"                  trans=\"{subtrans[i][0]}\"/>\n"
                        f"                </instance>\n"
                    )
                else:
                    self.trans += (
                        f"        <instance id=\"{trans_id}\"\n"
                        f"                  trans=\"{subtrans[i][0]}\">\n"
                    )

    def get_instance_id(self):
        return self.instanceID

    def set_instance_id(self, instance_id_list):
        self.instanceID = instance_id_list

    def __str__(self):
        if self.trans is not None:
            return (
                f"      <instance id=\"{self.id}\"\n"
                f"                page=\"{self.page}\">\n"
                f"{self.trans}"
                f"      </instance>\n"
            )
        else:
            return (
                f"      <instance id=\"{self.id}\"\n"
                f"                page=\"{self.page}\">\n"
                f"      </instance>\n"
            )
