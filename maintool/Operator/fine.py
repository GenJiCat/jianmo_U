# fine.py
# 细粒度算子模板库 (Fine-grained Operator Templates) - 支持子页面端口

# fine.py
# 细粒度算子模板库 (Fine-grained Operator Templates) - 支持子页面端口

def add_MatMul(cm, xml_list, prev_place_id, start_x, start_y, layer_id, op_id,
               weight_val="",
               is_in_port=False, in_port_name="In",
               is_out_port=False, out_port_name="Out",
               is_weight_port=True, weight_port_name="Weight"):  # 【新增】权重端口控制
    """
    添加矩阵乘法算子 (MatMul)
    """

    gap_x = 250
    gap_y = 150
    # prefix = f"L{layer_id}_{op_id}" # 未使用，可注释
    current_x = start_x
    current_y = start_y

    created_ports = []

    # 1. 输入库所 (Input Place)
    if prev_place_id:
        input_place_id = prev_place_id
    else:
        if is_in_port:
            res_in = cm.addPortPlace(current_x, current_y, "LR", "In")
            created_ports.append({'id': res_in[1], 'name': in_port_name, 'port_type': 'In'})
        else:
            res_in = cm.addPlace(current_x, current_y, "LR")
        xml_list.append(res_in[0])
        input_place_id = res_in[1]

    # 2. 权重库所 (Weight Place) - 【修改部分】
    w_x = current_x + gap_x * 0.5
    w_y = current_y - gap_y

    if is_weight_port:
        # 创建为输入端口，用于从主页面接收参数
        res_w = cm.addPortPlace(w_x, w_y, "LLR", "In")
        created_ports.append({'id': res_w[1], 'name': weight_port_name, 'port_type': 'In'})
    else:
        # 创建为本地带初始值的库所
        res_w = cm.addPlace_init(w_x, w_y, "LLR", initmark=weight_val, black=False)

    xml_list.append(res_w[0])
    weight_place_id = res_w[1]

    # 3. 变迁 (Transition)
    t_x = current_x + gap_x
    res_t = cm.addTrance(t_x, current_y, black=False)
    xml_list.append(res_t[0])
    trans_id = res_t[1]

    # 4. 输出库所 (Output Place)
    out_x = t_x + gap_x
    if is_out_port:
        res_out = cm.addPortPlace(out_x, current_y, "LR", "Out")
        created_ports.append({'id': res_out[1], 'name': out_port_name, 'port_type': 'Out'})
    else:
        res_out = cm.addPlace(out_x, current_y, "LR", black=False)
    xml_list.append(res_out[0])
    output_place_id = res_out[1]

    # 5. 弧 (Arcs)
    # Input -> Trans
    xml_list.append(cm.addArc(trans_id, input_place_id, "PtoT", "x",
                              ax=(current_x + t_x) / 2, ay=current_y + 10))
    # Weight -> Trans
    xml_list.append(cm.addArc(trans_id, weight_place_id, "PtoT", "w",
                              ax=w_x, ay=(w_y + current_y) / 2))
    # Trans -> Output
    xml_list.append(cm.addArc(trans_id, output_place_id, "TtoP", "mult(x, w)",
                              ax=(t_x + out_x) / 2, ay=current_y - 10))

    return output_place_id, out_x, current_y, created_ports


def add_BiasAdd(cm, xml_list, prev_place_id, start_x, start_y, layer_id, op_id,
                bias_val="",
                is_out_port=False, out_port_name="Out",
                is_bias_port=True, bias_port_name="Bias"):  # 【新增】偏置端口控制
    """
    偏置加法
    """
    gap_x = 250
    gap_y = 150
    current_x = start_x
    created_ports = []

    # 1. 输入处理
    if not prev_place_id:
        res_in = cm.addPlace(current_x, start_y, "LR", black=False)
        xml_list.append(res_in[0])
        prev_place_id = res_in[1]

    # 2. 偏置库所 - 【修改部分】
    b_x = current_x + gap_x * 0.5
    b_y = start_y - gap_y

    if is_bias_port:
        # 创建为输入端口
        res_b = cm.addPortPlace(b_x, b_y, "LR", "In")
        created_ports.append({'id': res_b[1], 'name': bias_port_name, 'port_type': 'In'})
    else:
        # 本地库所
        res_b = cm.addPlace_init(b_x, b_y, "LR", initmark=bias_val, black=False)

    xml_list.append(res_b[0])
    bias_place_id = res_b[1]

    # 3. 变迁
    t_x = current_x + gap_x
    res_t = cm.addTrance(t_x, start_y, black=False)
    xml_list.append(res_t[0])
    trans_id = res_t[1]

    # 4. 输出处理
    out_x = t_x + gap_x
    if is_out_port:
        res_out = cm.addPortPlace(out_x, start_y, "LR", "Out")
        created_ports.append({'id': res_out[1], 'name': out_port_name, 'port_type': 'Out'})
    else:
        res_out = cm.addPlace(out_x, start_y, "LR", black=False)
    xml_list.append(res_out[0])
    output_place_id = res_out[1]

    # 5. 弧
    xml_list.append(cm.addArc(trans_id, prev_place_id, "PtoT", "x", ax=(current_x + t_x) / 2, ay=start_y + 10))
    xml_list.append(cm.addArc(trans_id, bias_place_id, "PtoT", "b", ax=b_x, ay=(b_y + start_y) / 2))
    xml_list.append(
        cm.addArc(trans_id, output_place_id, "TtoP", "vec_add(x, b)", ax=(t_x + out_x) / 2, ay=start_y - 10))

    return output_place_id, out_x, start_y, created_ports


def add_ReLU(cm, xml_list, prev_place_id, start_x, start_y, layer_id, op_id,
             is_out_port=False, out_port_name="Out"):
    """
    ReLU 激活函数 (无参数，无需修改端口逻辑)
    """
    return add_Activation(cm, xml_list, prev_place_id, start_x, start_y, layer_id, op_id,
                          "ReLU", "apply_relu", False, "", is_out_port, out_port_name)


def add_Activation(cm, xml_list, prev_place_id, start_x, start_y, layer_id, op_id,
                   act_name="ReLU", ml_func="apply_relu",
                   is_in_port=False, in_port_name="In",
                   is_out_port=False, out_port_name="Out"):
    """
    通用单输入单输出算子 (无参数，无需修改端口逻辑)
    """
    gap_x = 200
    current_x = start_x
    current_y = start_y
    created_ports = []

    # 1. 输入处理
    if prev_place_id:
        input_place_id = prev_place_id
    else:
        if is_in_port:
            res_in = cm.addPortPlace(current_x, current_y, "LR", "In")
            created_ports.append({'id': res_in[1], 'name': in_port_name, 'port_type': 'In'})
        else:
            res_in = cm.addPlace(current_x, current_y, "LR", black=False)
        xml_list.append(res_in[0])
        input_place_id = res_in[1]

    # 2. 变迁
    t_x = current_x + gap_x * 0.5
    res_t = cm.addTrance(t_x, current_y, black=False)
    xml_list.append(res_t[0])
    trans_id = res_t[1]

    # 3. 输出处理
    out_x = t_x + gap_x * 0.5
    if is_out_port:
        res_out = cm.addPortPlace(out_x, current_y, "LR", "Out")
        created_ports.append({'id': res_out[1], 'name': out_port_name, 'port_type': 'Out'})
    else:
        res_out = cm.addPlace(out_x, current_y, "LR", black=False)
    xml_list.append(res_out[0])
    output_place_id = res_out[1]

    # 4. 弧
    xml_list.append(cm.addArc(trans_id, input_place_id, "PtoT", "x",
                              ax=(current_x + t_x) / 2, ay=current_y + 10))
    arc_expr = f"{ml_func}(x)"
    xml_list.append(cm.addArc(trans_id, output_place_id, "TtoP", arc_expr,
                              ax=(t_x + out_x) / 2, ay=current_y - 10))

    return output_place_id, out_x, current_y, created_ports

