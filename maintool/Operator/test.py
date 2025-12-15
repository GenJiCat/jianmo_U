import coarse
from maintool.createModel import CreatModel


# ==========================================
# 1. 注入全局定义
# ==========================================
def inject_globals(cm):
    cm.addColor(('list', 'LR', 'REAL'))
    cm.addColor(('list', 'LLR', 'LR'))
    cm.addMl("colset WB_PAIR = product LLR * LR;")
    cm.addVar("LR", "x")
    cm.addVar("LR", "y")

    # 函数桩
    ml_stub = """
    fun mult(v, m) = v;
    fun vec_add(v1, v2) = v1;
    fun apply_relu(v) = v;
    fun apply_softmax(v) = v;
    """
    cm.addMl(ml_stub)


# ==========================================
# 2. 构建子页面 (修正版)
# ==========================================
def build_2layer_block(cm, start_layer_index):
    xml_list = []
    ports_info = []

    curr_x, curr_y = 0.0, 0.0
    last_id = None

    # Layer N
    out_id, curr_x, curr_y, p1 = coarse.add_CoarseDense(
        cm, xml_list,
        prev_place_id=None,
        start_x=curr_x, start_y=curr_y,
        layer_id=start_layer_index,
        wb_init_val="1`(([[1.0]],[0.1]))",
        activation="relu",
        is_in_port=True, in_port_name="BlockIn",
        is_out_port=False
    )
    ports_info.extend(p1)
    last_id = out_id

    # Layer N+1
    out_id, curr_x, curr_y, p2 = coarse.add_CoarseDense(
        cm, xml_list,
        prev_place_id=last_id,
        start_x=curr_x, start_y=curr_y,
        layer_id=start_layer_index + 1,
        wb_init_val="1`(([[1.0]],[0.1]))",
        activation="relu",
        is_in_port=False,
        is_out_port=True, out_port_name="BlockOut"
    )
    ports_info.extend(p2)

    # 【修正点 1】：instance=False
    # 子页面不作为独立的顶层 Instance，而是依附于主页面的替代变迁
    page_id = cm.addPage(xml_list, subtrans=[], instance=False)

    return page_id, ports_info


# ==========================================
# 3. 构建主页面 (修正版)
# ==========================================
def build_main_hierarchical(cm, cpntxt: list):
    cm.setCpntxt(cpntxt)
    inject_globals(cm)

    main_xml_list = []

    start_x, start_y = -400.0, 0.0
    gap_x = 300.0

    res_sock = cm.addPlace(start_x, start_y, "LR", black=False)
    main_xml_list.append(res_sock[0])
    current_socket_id = res_sock[1]

    current_x = start_x

    # 收集主页面内所有的替代变迁ID
    all_substitution_trans_ids = []

    for i in range(3):
        layer_start = i * 2 + 1

        # A. 构建子页面
        sub_page_id, sub_ports = build_2layer_block(cm, layer_start)

        # B. 下一个槽口
        next_x = current_x + gap_x
        res_next_sock = cm.addPlace(next_x, start_y, "LR", black=False)
        main_xml_list.append(res_next_sock[0])
        next_socket_id = res_next_sock[1]

        # C. 绑定
        port_sock_str = ""
        for port in sub_ports:
            if port['name'] == "BlockIn":
                port_sock_str += f"({port['id']},{current_socket_id})"
            elif port['name'] == "BlockOut":
                port_sock_str += f"({port['id']},{next_socket_id})"

        # D. 替代变迁
        trans_x = (current_x + next_x) / 2
        block_name = f"Block_{i + 1}"

        res_sub = cm.addSubTrance(trans_x, start_y, sub_page_id, port_sock_str, block_name)
        main_xml_list.append(res_sub[0])
        sub_trans_id = res_sub[1]

        # 收集 ID
        all_substitution_trans_ids.append(sub_trans_id)

        # E. 弧
        main_xml_list.append(
            cm.addArc(sub_trans_id, current_socket_id, "PtoT", "", ax=(current_x + trans_x) / 2, ay=start_y))
        main_xml_list.append(cm.addArc(sub_trans_id, next_socket_id, "TtoP", "", ax=(trans_x + next_x) / 2, ay=start_y))

        current_socket_id = next_socket_id
        current_x = next_x

    # 【修正点 2】：传入 [all_substitution_trans_ids] (嵌套列表)
    # 你的 CreatModel 期望 subtrans 是一个 list of lists，或者处理逻辑需要这样包裹
    # 对应 mlp_to_cpn.py 中的 model.addPage(elems, [subtrans], True)
    cm.addPage(main_xml_list, subtrans=[all_substitution_trans_ids], instance=True)

    print("Main Hierarchical Page Created.")


# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    input_file = "../initial.cpn"
    output_file = "test/hierarchical_6_layers_fixed.cpn"

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
    except FileNotFoundError:
        print(f"Error: 找不到模板文件 {input_file}")
        exit(1)

    model = CreatModel()

    try:
        build_main_hierarchical(model, lines)

        # 确保输出目录存在
        import os

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(model.getCpntxt()))
        print(f"Success! Model saved to {output_file}")
    except Exception as e:
        import traceback

        traceback.print_exc()