def add_CoarseDense(cm, xml_list, prev_place_id, start_x, start_y, layer_id,
                    wb_init_val="",  
                    activation="relu",
                    is_in_port=False, in_port_name="In",
                    is_out_port=False, out_port_name="Out",
                    is_wb_port=True, wb_port_name="WB_Params"):  # 修改默认值为 True
    """
    粗粒度全连接层算子
    [Input] -> (x) -> [Trans] -> [Output]
                        ^
                        | (WB Port)
    """

    gap_x = 250
    gap_y = 150
    current_x = start_x
    current_y = start_y
    created_ports = []

    colset_io = "LR"
    colset_wb = "WB_PAIR"

    # 1. 输入库所
    if prev_place_id:
        input_place_id = prev_place_id
    else:
        if is_in_port:
            res_in = cm.addPortPlace(current_x, current_y, colset_io, "In")
            created_ports.append({'id': res_in[1], 'name': in_port_name, 'port_type': 'In'})
        else:
            res_in = cm.addPlace(current_x, current_y, colset_io, black=False)
        xml_list.append(res_in[0])
        input_place_id = res_in[1]

    # 2. 权重偏置复合库所 (WB Place) - 这里的逻辑升级了
    wb_x = current_x + gap_x * 0.5
    wb_y = current_y - gap_y

    if is_wb_port:
        # 强制生成为端口，不再设置初始 Token (由主页面传入)
        # 端口类型必须是 "In"，因为是参数输入
        res_wb = cm.addPortPlace(wb_x, wb_y, colset_wb, "In")
        created_ports.append({'id': res_wb[1], 'name': wb_port_name, 'port_type': 'In'})
    else:
        res_wb = cm.addPlace_init(wb_x, wb_y, colset_wb, initmark=wb_init_val, black=False)

    xml_list.append(res_wb[0])
    wb_place_id = res_wb[1]

    # 3. 变迁
    t_x = current_x + gap_x
    res_t = cm.addTrance(t_x, current_y, black=False)
    xml_list.append(res_t[0])
    trans_id = res_t[1]

    # 4. 输出库所
    out_x = t_x + gap_x
    if is_out_port:
        res_out = cm.addPortPlace(out_x, current_y, colset_io, "Out")
        created_ports.append({'id': res_out[1], 'name': out_port_name, 'port_type': 'Out'})
    else:
        res_out = cm.addPlace(out_x, current_y, colset_io, black=False)
    xml_list.append(res_out[0])
    output_place_id = res_out[1]

    # 5. 弧
    xml_list.append(cm.addArc(trans_id, input_place_id, "PtoT", "x",
                              ax=(current_x + t_x) / 2, ay=current_y + 10))

    xml_list.append(cm.addArc(trans_id, wb_place_id, "PtoT", "(w, b)",
                              ax=wb_x, ay=(wb_y + current_y) / 2))

    core_calc = "vec_add(mult(x, w), b)"
    if activation.lower() == "relu":
        final_expr = f"apply_relu({core_calc})"
    elif activation.lower() == "sigmoid":
        final_expr = f"apply_sigmoid({core_calc})"
    elif activation.lower() == "tanh":
        final_expr = f"apply_tanh({core_calc})"
    elif activation.lower() == "softmax":
        final_expr = f"apply_softmax({core_calc})"
    else:
        final_expr = core_calc

    xml_list.append(cm.addArc(trans_id, output_place_id, "TtoP", final_expr,
                              ax=(t_x + out_x) / 2, ay=current_y - 10))

    return output_place_id, out_x, current_y, created_ports