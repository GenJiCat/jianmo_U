import json
import os
from Operator import fine
from Operator import coarse
from maintool.createModel import CreatModel


class AutoCPNBuilder:
    def __init__(self, creat_model: CreatModel):
        self.cm = creat_model
        self.curr_x = 0.0
        self.curr_y = 0.0

    def _reset_layout(self):
        self.curr_x = 0.0
        self.curr_y = 0.0

    def inject_globals(self, tree_nodes):
        """注入全局定义"""
        # 1. 基础类型
        self.cm.addColor(('list', 'LR', 'REAL'))
        self.cm.addColor(('list', 'LLR', 'LR'))
        self.cm.addColor('UNIT')
        self.cm.addMl("colset WB_PAIR = product LLR * LR;")

        self.cm.addVar("LR", "x")
        self.cm.addVar("LR", "b")
        self.cm.addVar("LLR", "w")
        self.cm.addVar("UNIT", "u")

        # 2. 函数桩
        ml_code = """
        fun mult(x, w) = x;
        fun vec_add(a, b) = a;
        fun apply_relu(x) = x;
        fun apply_sigmoid(x) = x;
        fun apply_softmax(x) = x;
        fun flatten(x) = x;
        """
        self.cm.addMl(ml_code)

        # 3. 动态生成参数引用 (GlobRef)
        # 【新增】输入数据的全局引用
        self.cm.addMl("globref ref_input_x = [0.0];")

        # 各层权重的引用
        for node in tree_nodes:
            if node['type'] == 'CoarseBlock':
                layer_name = node['source_layer']
                self.cm.addMl(f"globref ref_w_{layer_name} = [[0.0]];")
                self.cm.addMl(f"globref ref_b_{layer_name} = [0.0];")

    def _format_init_val(self, params, mode="coarse"):
        if mode == "coarse":
            return "1`(([[1.0]], [0.1]))"
        elif mode == "weight":
            return "1`[[1.0]]"
        elif mode == "bias":
            return "1`[0.1]"
        return ""

    def build_hierarchical_model(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tree = data.get("structure_tree", [])
        model_name = data['meta']['model_name']
        print(f"Building: {model_name}")

        self.inject_globals(tree)

        subpage_registry = []

        # --- 1. 生成子页面 ---
        for node in tree:
            node_type = node.get('type')
            layer_name = node.get('source_layer')
            print(f"  > Subpage: {layer_name} ({node_type})")

            page_info = None
            if node_type == 'CoarseBlock':
                page_info = self._build_subpage_coarse(node)
            elif node_type == 'FineBlock':
                # 假设 FineBlock 内部不需要主页面分发参数（或者逻辑不同），这里暂时略过
                page_info = self._build_subpage_fine_group(node.get('children', []), layer_name)
            elif node_type == 'AtomicOp':
                page_info = self._build_subpage_fine_group([node], layer_name)

            if page_info:
                page_info['params'] = node.get('params', {})
                subpage_registry.append(page_info)

        # --- 2. 生成主页面 ---
        self._build_main_page(subpage_registry)
        print("Done.")

    # ... [子页面构建代码与之前相同，此处省略以节省篇幅] ...
    # 请直接保留原来的 _build_subpage_coarse 和 _build_subpage_fine_group

    def _build_subpage_coarse(self, node):
        self._reset_layout()
        xml_list = []
        ports_created = []
        _, _, _, ports = coarse.add_CoarseDense(
            self.cm, xml_list, None, self.curr_x, self.curr_y,
            node['node_id'], "", node['params'].get("activation", "none"),
            True, "In", True, "Out", True, "WB"  # 开启 WB 端口
        )
        ports_created.extend(ports)
        page_id = self.cm.addPage(xml_list, subtrans=[], instance=False)
        return self._extract_page_info(page_id, ports_created, node['source_layer'])

    def _build_subpage_fine_group(self, ops, layer_name):
        self._reset_layout()
        xml_list = []
        ports_created = []
        last_id = None
        for j, op in enumerate(ops):
            is_first = (j == 0)
            is_last = (j == len(ops) - 1)
            out_id, nx, ny, ports = self._dispatch_fine_op(
                op, xml_list, last_id, is_first, "In" if is_first else "", is_last, "Out" if is_last else ""
            )
            if ports: ports_created.extend(ports)
            last_id = out_id
            self.curr_x = nx
            self.curr_y = ny
        page_id = self.cm.addPage(xml_list, subtrans=[], instance=False)
        return self._extract_page_info(page_id, ports_created, layer_name)




    def _dispatch_fine_op(self, op, xml_list, prev_id, is_in, in_name, is_out, out_name):
        """
        分发细粒度算子，并开启参数端口
        """
        t_name = op.get("template_name") or op.get("op_type")
        params = op.get("params", {})
        op_name = f"{op.get('source_layer')}_{t_name}"

        if t_name == "MatMul":
            return fine.add_MatMul(
                self.cm, xml_list, prev_id, self.curr_x, self.curr_y,
                op_name, 0,
                self._format_init_val(params, "weight"),
                is_in, in_name, is_out, out_name,
                is_weight_port=True, weight_port_name="Weight"  # 【新增】开启端口
            )
        elif t_name == "BiasAdd":
            return fine.add_BiasAdd(
                self.cm, xml_list, prev_id, self.curr_x, self.curr_y,
                op_name, 0,
                self._format_init_val(params, "bias"),
                is_out, out_name,
                is_bias_port=True, bias_port_name="Bias"  # 【新增】开启端口
            )
        elif t_name == "Activation":
            func = params.get("func_name", "relu")
            return fine.add_Activation(
                self.cm, xml_list, prev_id, self.curr_x, self.curr_y,
                op_name, 0,
                act_name=func, ml_func=f"apply_{func}",
                is_out_port=is_out, out_port_name=out_name
            )
        return prev_id, self.curr_x, self.curr_y, []

    def _extract_page_info(self, page_id, ports_created, name):
        """
        提取子页面关键端口信息
        """
        in_id = next((p['id'] for p in ports_created if p['name'] == "In"), None)
        out_id = next((p['id'] for p in ports_created if p['name'] == "Out"), None)

        # 粗粒度端口
        wb_id = next((p['id'] for p in ports_created if p['name'] == "WB"), None)

        # 细粒度端口 (细粒度页面可能有独立的 Weight 和 Bias 端口)
        # 注意：如果一层有多个 MatMul，可能会有多个 Weight 端口，这里简化假设每层一个
        weight_id = next((p['id'] for p in ports_created if p['name'] == "Weight"), None)
        bias_id = next((p['id'] for p in ports_created if p['name'] == "Bias"), None)

        return {
            'page_id': page_id,
            'in_port': in_id,
            'out_port': out_id,
            'wb_port': wb_id,  # Coarse 用
            'weight_port': weight_id,  # Fine 用
            'bias_port': bias_id,  # Fine 用
            'name': name
        }

    # =========================================================================
    # 主页面构建器 (适配细粒度参数)
    # =========================================================================

    def _build_main_page(self, subpages):
        self._reset_layout()
        main_xml = []

        start_x = -400.0
        gap_x = 350.0
        param_y = 200.0

        # 1. Loader 部分 (代码同前，略微调整参数槽口连接)
        # ... (Loader创建代码不变) ...
        loader_x = start_x
        loader_y = param_y + 150
        res_start = self.cm.addPlace_init(loader_x, loader_y, "UNIT", "1`()", black=False)
        main_xml.append(res_start[0])
        start_id = res_start[1]

        read_lines = [f'ref_input_x := [1.0, 2.0];']
        for sp in subpages:
            if 'params' in sp:
                ln = sp['name']
                # 无论粗细粒度，都生成读文件代码 (简化模拟)
                if sp['params'].get('weight_file'): read_lines.append(f'ref_w_{ln} := [[1.0]];')
                if sp['params'].get('bias_file'): read_lines.append(f'ref_b_{ln} := [0.1];')

        loader_code = "action\nlet\n" + "\n".join(read_lines) + "\nin\n  ()\nend;"
        res_loader = self.cm.addTrance_file(loader_x + 150, loader_y, loader_code)
        main_xml.append(res_loader[0])
        loader_id = res_loader[1]

        main_xml.append(self.cm.addArc(loader_id, start_id, "PtoT", "u", loader_x + 75, loader_y))

        # 2. 主链条
        self.curr_x = start_x
        res_sock = self.cm.addPlace(self.curr_x, 0, "LR", black=False)
        main_xml.append(res_sock[0])
        curr_sock_id = res_sock[1]

        main_xml.append(
            self.cm.addArc(loader_id, curr_sock_id, "TtoP", "!ref_input_x", self.curr_x, (loader_y + 0) / 2))

        all_sub_trans = []

        for sp in subpages:
            ln = sp['name']

            # --- 参数槽口逻辑 (适配 Coarse 和 Fine) ---

            # 2.1 粗粒度 WB 端口
            if sp.get('wb_port'):
                p_sock_x = self.curr_x + gap_x / 2
                res_wb = self.cm.addPlace(p_sock_x, param_y, "WB_PAIR", black=False)
                main_xml.append(res_wb[0])
                sp['wb_socket_id'] = res_wb[1]

                # Loader -> Socket
                annot = f"(!ref_w_{ln}, !ref_b_{ln})"
                main_xml.append(self.cm.addArc(loader_id, res_wb[1], "TtoP", annot, p_sock_x, (loader_y + param_y) / 2))

            # 2.2 细粒度 Weight 端口
            if sp.get('weight_port'):
                p_sock_x = self.curr_x + gap_x / 3  # 稍微错开位置
                res_w = self.cm.addPlace(p_sock_x, param_y + 50, "LLR", black=False)
                main_xml.append(res_w[0])
                sp['weight_socket_id'] = res_w[1]

                # Loader -> Socket
                main_xml.append(
                    self.cm.addArc(loader_id, res_w[1], "TtoP", f"!ref_w_{ln}", p_sock_x, (loader_y + param_y) / 2))

            # 2.3 细粒度 Bias 端口
            if sp.get('bias_port'):
                p_sock_x = self.curr_x + gap_x * 2 / 3
                res_b = self.cm.addPlace(p_sock_x, param_y - 50, "LR", black=False)
                main_xml.append(res_b[0])
                sp['bias_socket_id'] = res_b[1]

                # Loader -> Socket
                main_xml.append(
                    self.cm.addArc(loader_id, res_b[1], "TtoP", f"!ref_b_{ln}", p_sock_x, (loader_y + param_y) / 2))

            # --- 主链推进 ---
            next_x = self.curr_x + gap_x
            res_next = self.cm.addPlace(next_x, 0, "LR", black=False)
            main_xml.append(res_next[0])
            next_sock_id = res_next[1]

            # Binding
            binding = ""
            if sp['in_port']: binding += f"({sp['in_port']},{curr_sock_id})"
            if sp['out_port']: binding += f"({sp['out_port']},{next_sock_id})"
            if sp.get('wb_port'): binding += f"({sp['wb_port']},{sp['wb_socket_id']})"
            if sp.get('weight_port'): binding += f"({sp['weight_port']},{sp['weight_socket_id']})"
            if sp.get('bias_port'): binding += f"({sp['bias_port']},{sp['bias_socket_id']})"

            trans_x = (self.curr_x + next_x) / 2
            res_sub = self.cm.addSubTrance(trans_x, 0, sp['page_id'], binding, sp['name'])
            main_xml.append(res_sub[0])
            sub_id = res_sub[1]
            all_sub_trans.append(sub_id)

            # 视觉弧
            main_xml.append(self.cm.addArc(sub_id, curr_sock_id, "PtoT", "", (self.curr_x + trans_x) / 2, 0))
            main_xml.append(self.cm.addArc(sub_id, next_sock_id, "TtoP", "", (trans_x + next_x) / 2, 0))

            # 参数视觉弧
            if sp.get('wb_port'):
                main_xml.append(self.cm.addArc(sub_id, sp['wb_socket_id'], "PtoT", "", trans_x, param_y / 2))
            if sp.get('weight_port'):
                main_xml.append(self.cm.addArc(sub_id, sp['weight_socket_id'], "PtoT", "", trans_x, (param_y + 50) / 2))
            if sp.get('bias_port'):
                main_xml.append(self.cm.addArc(sub_id, sp['bias_socket_id'], "PtoT", "", trans_x, (param_y - 50) / 2))

            curr_sock_id = next_sock_id
            self.curr_x = next_x

        self.cm.addPage(main_xml, subtrans=[all_sub_trans], instance=True)


# 确保 main 里的 addArc 调用参数名对齐你的 createmodel.py
# 你的 addArc 定义: addArc(self, transend, placeend, orientation, text, ax, ay)
# 所以上述代码中的调用顺序调整为: model.addArc(loader_id, curr_sock_id, 'TtoP', 'text', x, y)


if __name__ == "__main__":
    json_path = "test/mlp2/reduced.json"
    output_path = "test/mlp2/auto_hierarchical_mixed.cpn"
    template_path = "initial.cpn"

    if not os.path.exists(json_path):
        print("JSON not found")
        exit()

    with open(template_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]

    model = CreatModel()
    model.setCpntxt(lines)

    builder = AutoCPNBuilder(model)
    builder.build_hierarchical_model(json_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(model.getCpntxt()))
    print(f"Generated: {output_path}")