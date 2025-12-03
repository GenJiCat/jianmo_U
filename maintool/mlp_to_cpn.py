#!/usr/bin/env python3
"""
mlp_to_cpn.py

在主页面的初始输入库所前添加“读取文件”库所+变迁；
在每个子页面的 input / weight / bias 三类库所前各添加“读取文件”库所+变迁。
对生成这些读取元素的位置在代码中做了注释，便于你后续修改。
"""

import os
import json
import argparse
from typing import Tuple, Dict, List

import sys
# allow importing createModel from parent dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from createModel import CreatModel


def build_layer_page(model: CreatModel,
                     layer_name: str,
                     in_features: int,
                     out_features: int,
                     idx,
                     x0: float,
                     y0: float,
                     dx: float = 120.0) -> Tuple[str, str, str]:
    """
    为单层构建子页面（水平布局），返回 (page_id, input_port_pid, output_port_pid)
    布局（从左到右）:
      input_port --> [weight_file -> weight] -> [bias_file -> bias] -> mid1 -> mid2 -> output
    说明：子页面的 input port 不再有前置的读取文件变迁/库所（由主页面传入数据）。
    权重和偏置前仍保留读取文件的库所与变迁。
    """
    elems: List[str] = []

    # left-to-right positions
    x_input = x0
    x_weight = x0 + dx
    x_bias = x0 + 2 * dx
    x_mid1 = x0 + 3 * dx
    x_mid2 = x0 + 4 * dx
    x_output = x0 + 5 * dx

    # Y positions
    y_place = y0
    y_trans = y0 - dx * 0.4

    # ---------------------------
    # 1) 子页面：输入（直接端口，不再添加读取文件结构）
    # ---------------------------
    # 直接创建 input port place (port type 'in')；主页面负责提供数据
    p_input_str, p_input_id, _, _ = model.addPortPlace(x_input, y_place, 'LR', 'in')
    elems.append(p_input_str)
    # ← 子页面输入端口库所（主页面传入，不需要子页面读取变迁）

    # ---------------------------
    # 2) 子页面：权重（带前置读取）
    # ---------------------------
    # 读取文件用的权重库所（weight file place）
    p_weight_file_str, p_weight_file_id, _, _ = model.addPlace_init(x_weight - dx * 0.6, y_place, 'E', initmark='c')
    elems.append(p_weight_file_str)
    # ← 读取文件的库所（子页面权重的前置库所）

    # 实际的权重库所
    p_weight_str, p_weight_id, _, _ = model.addPlace(x_weight, y_place, 'LLR')
    elems.append(p_weight_str)
    # ← 子页面权重库所（由读取变迁输出）

    # 权重的读取变迁
    tx_weight_file = (x_weight - dx * 0.6 + x_weight) / 2.0
    t_weight_file_str, t_weight_file_id, _, _ = model.addTrance_file(tx_weight_file, y_trans, f"""
    action
let
  val arq_ent = TextIO.openIn(
  "C:/Users/CC/Desktop/output2/net__{idx*2}__weight.txt"
  );
  val dados = ELEM2.input_ms(arq_ent);
in
  m2 := dados;
  TextIO.closeIn(arq_ent);
  ()
end;
    """)
    elems.append(t_weight_file_str)
    # ← 读取数据变迁（子页面权重的读取变迁）

    # 弧：weight_file -> weight_read_tran (PtoT)
    elems.append(model.addArc(t_weight_file_id, p_weight_file_id, 'PtoT', 'c', tx_weight_file - dx*0.2, y_place))
    # ← 读取文件库所 -> 权重读取变迁 的弧（PtoT）
    # 弧：weight_read_tran -> weight_place (TtoP)
    elems.append(model.addArc(t_weight_file_id, p_weight_id, 'TtoP', 'elementosIniciais2()', tx_weight_file + dx*0.2, y_place))
    # ← 权重读取变迁 -> 权重库所 的弧（TtoP）

    # ---------------------------
    # 3) 子页面：偏置（带前置读取）
    # ---------------------------
    # 读取文件用的偏置库所（bias file place）
    p_bias_file_str, p_bias_file_id, _, _ = model.addPlace_init(x_bias - dx * 0.6, y_place, 'E', initmark='c')
    elems.append(p_bias_file_str)
    # ← 读取文件的库所（子页面偏置的前置库所）

    # 实际的偏置库所
    p_bias_str, p_bias_id, _, _ = model.addPlace(x_bias, y_place, 'LR')
    elems.append(p_bias_str)
    # ← 子页面偏置库所（由读取变迁输出）

    # 偏置的读取变迁
    tx_bias_file = (x_bias - dx * 0.6 + x_bias) / 2.0
    t_bias_file_str, t_bias_file_id, _, _ = model.addTrance_file(tx_bias_file, y_trans, f"""
    action
let
  val arq_ent = TextIO.openIn(
  "C:/Users/CC/Desktop/output2/net__{idx*2}__bias.txt"
  );
  val dados = ELEM.input_ms(arq_ent);
in
  m0 := dados;
  TextIO.closeIn(arq_ent);
  ()
end;
    """)
    elems.append(t_bias_file_str)
    # ← 读取数据变迁（子页面偏置的读取变迁）

    # 弧：bias_file -> bias_read_tran (PtoT)
    elems.append(model.addArc(t_bias_file_id, p_bias_file_id, 'PtoT', 'c', tx_bias_file - dx*0.2, y_place))
    # ← 读取文件库所 -> 偏置读取变迁 的弧（PtoT）
    # 弧：bias_read_tran -> bias_place (TtoP)
    elems.append(model.addArc(t_bias_file_id, p_bias_id, 'TtoP', 'elementosIniciais()', tx_bias_file + dx*0.2, y_place))
    # ← 偏置读取变迁 -> 偏置库所 的弧（TtoP）

    # ---------------------------
    # 4) 子页面：中间和输出部分（原来逻辑）
    # ---------------------------
    # mid places
    p_mid1_str, p_mid1_id, _, _ = model.addPlace(x_mid1, y_place, 'LR')
    elems.append(p_mid1_str)
    p_mid2_str, p_mid2_id, _, _ = model.addPlace(x_mid2, y_place, 'LR')
    elems.append(p_mid2_str)

    # output port (port type 'out')
    p_out_str, p_out_id, _, _ = model.addPortPlace(x_output, y_place, 'LR', 'out')
    elems.append(p_out_str)

    # transitions (placed between corresponding places)
    # 1) weight multiply transition between input and weight (x_input <-> x_weight)
    tx1 = (x_input + x_weight) / 2.0
    t1_str, t1_id, _, _ = model.addTrance(tx1, y_trans)
    elems.append(t1_str)
    elems.append(model.addArc(t1_id, p_input_id, 'PtoT', 'x', tx1 - dx * 0.2, y_place))
    elems.append(model.addArc(t1_id, p_weight_id, 'TtoP', 'w', tx1 - dx*0.2, y_place))
    elems.append(model.addArc(t1_id, p_weight_id, 'PtoT', 'w', tx1 - dx * 0.2, y_place))
    elems.append(model.addArc(t1_id, p_mid1_id, 'TtoP', 'mult(w,x,nil)', tx1 + dx * 0.2, y_place))

    # 2) add bias transition between mid1 and bias/mid2 area
    tx2 = (x_mid1 + x_bias) / 2.0
    t2_str, t2_id, _, _ = model.addTrance(tx2, y_trans)
    elems.append(t2_str)
    elems.append(model.addArc(t2_id, p_mid1_id, 'PtoT', 'x', tx2 - dx * 0.2, y_place))
    elems.append(model.addArc(t2_id, p_bias_id, 'TtoP', 'b', tx2 - dx*0.2, y_place))
    elems.append(model.addArc(t2_id, p_bias_id, 'PtoT', 'b', tx2 - dx * 0.2, y_place))
    elems.append(model.addArc(t2_id, p_mid2_id, 'TtoP', 'vector_sum(x,b,nil)', tx2 + dx * 0.2, y_place))

    # 3) activation transition between mid2 and output
    tx3 = (x_mid2 + x_output) / 2.0
    t3_str, t3_id, _, _ = model.addTrance(tx3, y_trans)
    elems.append(t3_str)
    elems.append(model.addArc(t3_id, p_mid2_id, 'PtoT', 'x', tx3 - dx * 0.2, y_place))
    elems.append(model.addArc(t3_id, p_out_id, 'TtoP', 'relu_activation(x,nil)', tx3 + dx * 0.2, y_place))

    # create page
    page_id = model.addPage(elems, [], False)
    return page_id, p_input_id, p_out_id



def main(initial: str, structure: str, output: str):
    # load template
    with open(initial, 'r', encoding='utf-8') as f:
        template_lines = [l.rstrip('\\n') for l in f]

    # load mlp structure
    with open(structure, 'r', encoding='utf-8') as f:
        struct = json.load(f)

    # build list of actual MLP linear layers to model (prefer mlp_summary if present)
    mlp_layers_src = []
    mlp_summary = struct.get('mlp_summary')
    if mlp_summary and isinstance(mlp_summary.get('mlp_layers'), list):
        for item in mlp_summary['mlp_layers']:
            ln = item.get('layer_name')
            inf = item.get('in_features', 1)
            outf = item.get('out_features', 1)
            if ln:
                mlp_layers_src.append({'name': ln, 'in_features': inf, 'out_features': outf})
    else:
        for L in struct.get('layers', []):
            if L.get('type') == 'Linear':
                mlp_layers_src.append({
                    'name': L.get('name'),
                    'in_features': L.get('in_features', 1),
                    'out_features': L.get('out_features', 1)
                })

    if not mlp_layers_src:
        raise SystemExit("No Linear layers found to model in structure file.")

    model = CreatModel()
    model.setCpntxt(template_lines)

    # ensure color/vars and helper functions exist (your previous choices)
    model.addColor(('list', 'LR', 'REAL'))
    model.addColor(('list', 'LLR', 'LR'))
    model.addColor(('list', 'LI', 'INT'))  # 新增 BOOL 类型
    model.addVar("LR", 'x')
    model.addVar("LR", 'b')
    model.addVar("LLR", 'w')
    model.addVar("LI", 'y')  # 新增布尔变量 y

    # add ML helper functions (unchanged)
    model.addMl("""fun mult([],v,answer_vector) = rev(answer_vector)
     | mult(m,v,answer_vector)=
let
   val line = hd m

fun mlc([],[],result)=result
  | mlc(line,[],result) = result
  | mlc([],vector,result)=result
  | mlc(line,vector,result)=
    let
      val el = hd line
      val ev= hd vector
      val x = result + (el*ev)
    in
      mlc(tl line, tl vector, x)
    end

in
mult(tl m,v,mlc(line,v,0.0)::answer_vector)
end
""")
    model.addMl("""fun vector_sum ([], [], answer_vector: LR) = rev(answer_vector)
 | vector_sum (l, v, answer_vector: LR) =
     let
       val el = hd l
       val ev = hd v
     in
       vector_sum(tl l, tl v, (el + ev) :: answer_vector)
     end
""")
    model.addMl("""fun relu_activation ([], answer_vector) = rev(answer_vector)
 | relu_activation (v, answer_vector) =
     let
        val ui = hd v
        val activated = if ui > 0.0 then ui else 0.0
     in
        relu_activation (tl v, activated :: answer_vector)
     end
""")
    model.addMl("""
        fun elementosIniciais() = (!m0);
        """)
    model.addMl("""
        fun elementosIniciais2() = (!m2);
        """)
    model.addMl("""
    fun elementosIniciais3() = (!m3);
    """)
    model.addMl("""
   fun softmax (xs: real list) : real list =
  let
    val exps = List.map Math.exp xs
    val sumExps = List.foldl op+ 0.0 exps
  in
    List.map (fn x =&gt; x / sumExps) exps
  end;
    """)
    model.addMl("""fun argmax_opt (xs: real list) : int option =
  let
    fun loop (lst: real list, idx: int, best_i: int, best_v: real) =
      case lst of
         [] =&gt; best_i
       | h::t =&gt; if h &gt; best_v then loop(t, idx+1, idx, h) else loop(t, idx+1, best_i, best_v)
  in
    case xs of
       [] =&gt; NONE
     | h::t =&gt; SOME (loop(t, 1, 0, h))
  end
    """)
    model.addMl("""fun label_index (lbl: int list) : int option =
  let
    fun loop lst i =
      case lst of
         [] =&gt; NONE
       | h::t =&gt; if h &lt;&gt; 0 then SOME i else loop t (i+1)
  in
    loop lbl 0
  end;
    """)
    model.addMl("""fun PredictionC (probs: real list, label: int list) : bool =
  case argmax_opt probs of
     NONE =&gt; false
   | SOME pred =&gt;
       (case label_index label of
          NONE =&gt; false
        | SOME gt =&gt; pred = gt);
    """)
    model.addMl("""fun PredictionW(probs: real list, label: int list) : bool =
  not (PredictionC(probs, label));
    """)


    # Create child pages per MLP layer (avoid duplicates)
    page_info: Dict[str, Dict] = {}
    x0_base = 100.0
    y0_base = 120.0
    dx = 120.0
    horiz_spacing = 6 * dx + 80  # space between subpage blocks when placing them in overall workspace

    for idx, layer in enumerate(mlp_layers_src):
        print(idx)
        name = layer.get('name', f'layer{idx}')
        if name in page_info:
            continue
        in_f = layer.get('in_features', 1)
        out_f = layer.get('out_features', 1)
        x0 = x0_base + idx * horiz_spacing
        page_id, in_pid, out_pid = build_layer_page(model, name, in_f, out_f, idx, x0, y0_base, dx=dx)
        page_info[name] = {'page_id': page_id, 'in': in_pid, 'out': out_pid}

    # build main page: chain substitution transitions using the mlp_layers_src order
    elems: List[str] = []
    subtrans: List[str] = []

    # initial input place (main input) — 使用普通库所，并在前面加读取文件结构
    p_init_str, p_init_id, _, _ = model.addPlace(100, 300, 'LR')
    elems.append(p_init_str)
    # ← 这是主页面初始输入库所（目标库所，后续由读取变迁填充）

    # 添加主页面读取文件的库所和变迁
    p_file_str, p_file_id, _, _ = model.addPlace_init(100, 240, 'E', initmark="c")
    elems.append(p_file_str)
    # ← 读取文件的库所（主页面输入的前置库所）

    t_file_str, t_file_id, _, _ = model.addTrance_file(100, 260, f"""
    action
let
  val arq_ent = TextIO.openIn(
  "C:/Users/CC/Desktop/output2/x_test.txt"
  );
  val dados = ELEM.input_ms(arq_ent);
in
  m0 := dados;
  TextIO.closeIn(arq_ent);
  ()
end;""")
    elems.append(t_file_str)
    # ← 读取数据变迁（主页面输入的读取变迁）

    elems.append(model.addArc(t_file_id, p_file_id, 'PtoT', 'c', 100 - 20, 250))
    # ← 读取文件库所 -> 读取变迁 的弧（PtoT）
    elems.append(model.addArc(t_file_id, p_init_id, 'TtoP', 'elementosIniciais()', 100 + 20, 310))
    # ← 读取变迁 -> 主页面初始库所 的弧（TtoP）

    prev_pid = p_init_id
    for idx, layer in enumerate(mlp_layers_src):
        name = layer['name']
        info = page_info.get(name)
        if not info:
            continue

        # create return place (holds this layer's output in main page)
        p_ret_x = 200 + idx * 300
        p_ret_str, p_ret_id, _, _ = model.addPlace(p_ret_x, 400, 'LR')
        elems.append(p_ret_str)

        # build portsock; portsocket maps subpage ports to main page places
        portsock = f"({info['in']},{prev_pid})({info['out']},{p_ret_id})"
        x = p_ret_x
        t_sub_str, t_sub_id, _, _ = model.addSubTrance(x, 300, info['page_id'], portsock, name)
        elems.append(t_sub_str)

        # connect main prev place -> sub (PtoT)
        elems.append(model.addArc(t_sub_id, prev_pid, 'PtoT', 'x', x - 50, 300))
        # connect sub -> ret place (TtoP)
        elems.append(model.addArc(t_sub_id, p_ret_id, 'TtoP', 'x', x + 50, 300))

        subtrans.append(t_sub_id)
        prev_pid = p_ret_id

    # 5) Main page: Softmax activation and store result (新添加)
    num_layers = len(mlp_layers_src)
    x_last = 200 + (num_layers - 1) * 300
    p_softmax_str, p_softmax_id, _, _ = model.addPlace(x_last, 300, 'LR')
    elems.append(p_softmax_str)
    t_softmax_str, t_softmax_id, _, _ = model.addTrance(x_last, 340)
    elems.append(t_softmax_str)
    # 弧：最终输出 -> Softmax 变迁 (PtoT)
    elems.append(model.addArc(t_softmax_id, prev_pid, 'PtoT', 'x', x_last - 10, 370))
    # 弧：Softmax 变迁 -> Softmax 输出库所 (TtoP)
    elems.append(model.addArc(t_softmax_id, p_softmax_id, 'TtoP', 'softmax(x)', x_last + 10, 330))

    # 6) Main page: Y 读取和存储 (新添加)
    # 添加读取 y 文件的库所和变迁，以及 y 存储库所
    p_y_file_str, p_y_file_id, _, _ = model.addPlace_init(x_last, 180, 'E', initmark="c")
    elems.append(p_y_file_str)
    t_y_str, t_y_id, _, _ = model.addTrance_file(x_last, 220, f"""
   action
let
  val arq_ent = TextIO.openIn(
  "C:/Users/CC/Desktop/output2/y_test.txt"
  );
  val dados = ELEM3.input_ms(arq_ent);
in
  m3 := dados;
  TextIO.closeIn(arq_ent);
  ()
end;""")
    elems.append(t_y_str)
    # 弧：y 文件库所 -> y 读取变迁 (PtoT)
    elems.append(model.addArc(t_y_id, p_y_file_id, 'PtoT', 'c', x_last - 20, 200))
    # 添加实际的 y 存储库所（类型为 BOOL）
    p_y_str, p_y_id, _, _ = model.addPlace(x_last, 140, 'LI')
    elems.append(p_y_str)
    # 弧：y 读取变迁 -> y 存储库所 (TtoP)
    elems.append(model.addArc(t_y_id, p_y_id, 'TtoP', 'elementosIniciais3()', x_last + 20, 180))

    # 7) Main page: 验证预测结果是否正确 (新添加)
    # 添加验证变迁、正确库所和错误库所
    p_correct_str, p_correct_id, _, _ = model.addPlace(x_last+50, 400, 'BOOL')
    elems.append(p_correct_str)
    p_error_str, p_error_id, _, _ = model.addPlace(x_last+150, 400, 'BOOL')
    elems.append(p_error_str)
    t_check_str, t_check_id, _, _ = model.addTrance(x_last+100, 340)
    elems.append(t_check_str)
    # 弧：Softmax 输出和 y 库所 -> 验证变迁 (PtoT)
    elems.append(model.addArc(t_check_id, p_softmax_id, 'PtoT', 'x', x_last, 320))
    elems.append(model.addArc(t_check_id, p_y_id, 'PtoT', 'y', x_last, 320))
    # 弧：验证变迁 -> 正确库所 (TtoP)
    elems.append(model.addArc(t_check_id, p_correct_id, 'TtoP', 'PredictionC(x,y)', x_last+100, 350))
    # 弧：验证变迁 -> 错误库所 (TtoP)
    elems.append(model.addArc(t_check_id, p_error_id, 'TtoP', 'PredictionW(x,y)', x_last+100, 390))

    # add main page to model
    model.addPage(elems, [subtrans], True)

    # write output
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write("\n".join(model.getCpntxt()))
    print("Generated:", output)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert mlp_structure.json -> hierarchical CPNTools .cpn")
    parser.add_argument('--initial', required=True, help='initial template cpn file (initial.cpn)')
    parser.add_argument('--structure', required=True, help='mlp structure json (mlp_structure.json)')
    parser.add_argument('--output', default='output/mlp_model.cpn', help='output cpn file')
    args = parser.parse_args()
    main(args.initial, args.structure, args.output)
