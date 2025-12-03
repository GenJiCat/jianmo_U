#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp_to_cpn_train_step1_expanded_v2_io_ports_adjusted_with_file_read_and_x_snapshot.py

变更要点：
- 在主页面增加了一个统一的权重/偏置读文件源（p_w_src）和读取变迁（t_read_w）。
- t_read_w 有一个 PtoT 输入（来自 p_w_src），并且有多个 TtoP 输出指向主页面上每个权重库所(main_W)和每个偏置库所(main_B)。
- 在代码中用注释明确标注新增的库所/变迁/弧，便于后续调整。
"""

import os
import json
import argparse
from typing import Tuple, Dict, List

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from createModel import CreatModel


# =========================
# 子页面：前向（权重/偏置为端口；隐藏层3输出，输出层1输出）
# 子页面中权重/偏置端口为 'I/O'
# =========================
def build_fwd_layer_page_as_ports(model: CreatModel,
                                  layer_name: str,
                                  in_features: int,
                                  out_features: int,
                                  x0: float,
                                  y0: float,
                                  is_last: bool,
                                  dx: float = 120.0) -> Dict[str, str]:
    elems: List[str] = []

    # 位置
    x_input  = x0
    x_weight = x0 + dx
    x_bias   = x0 + 2*dx
    x_mid1   = x0 + 3*dx
    x_mid2   = x0 + 4*dx
    x_out    = x0 + 5*dx
    y_place  = y0
    y_trans  = y0 - dx*0.4

    # 端口：输入/权重/偏置（子页面中 W/B 为 I/O）
    pin_str, pin_id, _, _ = model.addPortPlace(x_input, y_place, 'LR',  'in')   # port: in
    elems.append(pin_str)
    pw_str, pw_id,   _, _ = model.addPortPlace(x_weight, y_place,'LLR','I/O')   # W (I/O)
    elems.append(pw_str)
    pb_str, pb_id,   _, _ = model.addPortPlace(x_bias,   y_place,'LR', 'I/O')   # b (I/O)
    elems.append(pb_str)

    # 中间库所
    p_mid1_str, p_mid1_id, _, _ = model.addPlace(x_mid1, y_place, 'LR')  # mid1
    elems.append(p_mid1_str)
    p_mid2_str, p_mid2_id, _, _ = model.addPlace(x_mid2, y_place, 'LR')  # mid2
    elems.append(p_mid2_str)

    # 输出端口（隐藏层三端口，输出层单端口）
    if not is_last:
        pout_str, pout_id, _, _ = model.addPortPlace(x_out, y_place, 'LR', 'out')          # out
        elems.append(pout_str)
        pout_snap_str, pout_snap_id, _, _ = model.addPortPlace(x_out, y_place+40, 'LR','out')  # out_snap
        elems.append(pout_snap_str)
        pout_pre_str, pout_pre_id, _, _ = model.addPortPlace(x_out, y_place+80, 'LR','out')    # out_pre
        elems.append(pout_pre_str)
    else:
        pout_str, pout_id, _, _ = model.addPortPlace(x_out, y_place, 'LR', 'out')  # out (output layer)
        elems.append(pout_str)
        pout_snap_id = None
        pout_pre_id  = None

    # 变迁：乘法（x * W） —— 用到 pw_id（I/O）作为读取端
    tx1 = (x_input + x_weight)/2.0
    t1_str, t1_id, _, _ = model.addTrance(tx1, y_trans)
    elems.append(t1_str)
    elems.append(model.addArc(t1_id, pin_id, 'PtoT', 'x', tx1-0.25*dx, y_place))     # in -> t1
    elems.append(model.addArc(t1_id, pw_id,  'PtoT', 'w', tx1-0.25*dx, y_place))     # W (I/O) -> t1 (PtoT)
    elems.append(model.addArc(t1_id, p_mid1_id, 'TtoP', 'mult(w,x,nil)', tx1+0.25*dx, y_place))   # t1 -> mid1

    # 变迁：加偏置 —— 用到 pb_id（I/O）作为读取端
    tx2 = (x_mid1 + x_bias)/2.0
    t2_str, t2_id, _, _ = model.addTrance(tx2, y_trans)
    elems.append(t2_str)
    elems.append(model.addArc(t2_id, p_mid1_id, 'PtoT', 'x', tx2-0.25*dx, y_place))   # mid1 -> t2
    elems.append(model.addArc(t2_id, pb_id,     'PtoT', 'b', tx2-0.25*dx, y_place))   # b (I/O) -> t2
    elems.append(model.addArc(t2_id, p_mid2_id, 'TtoP', '2`vector_sum(x,b,nil)', tx2+0.25*dx, y_place))   # t2 -> mid2

    if not is_last:
        # 激活 -> 输出/备份
        tx3 = (x_mid2 + x_out)/2.0
        t3_str, t3_id, _, _ = model.addTrance(tx3, y_trans)
        elems.append(t3_str)
        elems.append(model.addArc(t3_id, p_mid2_id, 'PtoT', 'x', tx3-0.25*dx, y_place))     # mid2 -> t3
        elems.append(model.addArc(t3_id, pout_id,      'TtoP', 'relu_activation(x,nil)', tx3+0.25*dx, y_place))   # t3 -> out
        elems.append(model.addArc(t3_id, pout_snap_id, 'TtoP', 'relu_activation(x,nil)', tx3+0.25*dx, y_place+40))# t3 -> out_snap

        # 直通未激活 -> out_pre
        tx4 = tx3
        t4_str, t4_id, _, _ = model.addTrance(tx4, y_trans+40)
        elems.append(t4_str)
        elems.append(model.addArc(t4_id, p_mid2_id, 'PtoT', 'x', tx4-0.25*dx, y_place+40))   # mid2 -> t4
        elems.append(model.addArc(t4_id, pout_pre_id,'TtoP', 'x', tx4+0.25*dx, y_place+80))    # t4 -> out_pre
    else:
        tx3 = (x_mid2 + x_out)/2.0
        t3_str, t3_id, _, _ = model.addTrance(tx3, y_trans)
        elems.append(t3_str)
        elems.append(model.addArc(t3_id, p_mid2_id, 'PtoT', 'x', tx3-0.25*dx, y_place))       # mid2 -> t3
        elems.append(model.addArc(t3_id, pout_id,   'TtoP', 'x', tx3+0.25*dx, y_place))       # t3 -> out

    page_id = model.addPage(elems, [], False)
    ret = {
        'page_id': page_id,
        'port_in': pin_id,
        'port_w':  pw_id,
        'port_b':  pb_id,
        'port_out': pout_id
    }
    if not is_last:
        ret['port_out_snap'] = pout_snap_id
        ret['port_out_pre']  = pout_pre_id
    return ret


# =========================
# 子页面：反向（输出层）
# 子页面中 W/B 端口为 'I/O'（允许被主页面读写）
# =========================
def build_back_last_layer_page(model: CreatModel,
                               layer_name: str,
                               x0: float,
                               y0: float,
                               dx: float = 120.0) -> Dict[str, str]:
    elems: List[str] = []

    # 端口
    p_loss_str, p_loss_id, _, _ = model.addPortPlace(x0, y0, 'LR', 'in')     # loss_in (in)
    elems.append(p_loss_str)
    p_w_str, p_w_id, _, _ = model.addPortPlace(x0+dx, y0, 'LLR', 'I/O')      # W_self (I/O)
    elems.append(p_w_str)
    p_b_str, p_b_id, _, _ = model.addPortPlace(x0+2*dx, y0, 'LR', 'I/O')     # B_self (I/O)
    elems.append(p_b_str)
    p_prev_str, p_prev_id, _, _ = model.addPortPlace(x0, y0+60, 'LR', 'in')  # prev_out_snapshot (in)
    elems.append(p_prev_str)

    # 中间库所
    p_loss_storage_str, p_loss_storage_id, _, _ = model.addPlace(x0+dx, y0+60, 'LR')  # loss_storage
    elems.append(p_loss_storage_str)
    p_outer_str, p_outer_id, _, _ = model.addPlace(x0+1.5*dx, y0+60, 'LLR')          # outer
    elems.append(p_outer_str)

    # 输出误差端口（out）
    p_out_loss_str, p_out_loss_id, _, _ = model.addPortPlace(x0+2*dx, y0+60, 'LR', 'out')  # out_loss (out)
    elems.append(p_out_loss_str)

    # 变迁1：prev_out + loss -> outer + loss_storage + out_loss
    t1_str, t1_id, _, _ = model.addTrance(x0+0.5*dx, y0+30)
    elems.append(t1_str)
    elems.append(model.addArc(t1_id, p_prev_id, 'PtoT', 'x', x0+0.3*dx, y0+60))        # prev -> t1
    elems.append(model.addArc(t1_id, p_loss_id, 'PtoT', 'l', x0+0.2*dx, y0))           # loss -> t1
    elems.append(model.addArc(t1_id, p_outer_id,'TtoP', 'outerProduct(l,x)', x0+1.2*dx, y0+60))        # t1 -> outer
    elems.append(model.addArc(t1_id, p_loss_storage_id, 'TtoP', 'l', x0+dx, y0+60))    # t1 -> loss_storage
    elems.append(model.addArc(t1_id, p_out_loss_id, 'TtoP', 'l', x0+2*dx, y0+60))      # t1 -> out_loss

    # 变迁2：outer + W_self -> W_self（回写）
    t2_str, t2_id, _, _ = model.addTrance(x0+1.7*dx, y0+30)
    elems.append(t2_str)
    elems.append(model.addArc(t2_id, p_outer_id, 'PtoT', 'w2', x0+1.4*dx, y0+60))      # outer -> t2
    elems.append(model.addArc(t2_id, p_w_id,     'PtoT', 'w1', x0+1.4*dx, y0))          # W_self (I/O) -> t2 (read)
    elems.append(model.addArc(t2_id, p_w_id,     'TtoP', '2`updateWeights(w1,w2,0.2)', x0+2.0*dx, y0))          # t2 -> W_self (I/O) 写回

    # 变迁3：loss_storage + B_self -> B_self（回写）
    t3_str, t3_id, _, _ = model.addTrance(x0+2.5*dx, y0+30)
    elems.append(t3_str)
    elems.append(model.addArc(t3_id, p_loss_storage_id, 'PtoT', 'l', x0+2*dx, y0+60))  # loss_storage -> t3
    elems.append(model.addArc(t3_id, p_b_id,             'PtoT', 'b', x0+2*dx, y0))      # B_self (I/O) -> t3 (read)
    elems.append(model.addArc(t3_id, p_b_id,             'TtoP', '2`updateBias(b,l,0.2)', x0+2.6*dx, y0))    # t3 -> B_self (I/O) 写回

    page_id = model.addPage(elems, [], False)
    return {
        'page_id': page_id,
        'port_loss': p_loss_id,
        'port_w':    p_w_id,
        'port_b':    p_b_id,
        'port_prev': p_prev_id,
        'port_out_loss': p_out_loss_id
    }


# =========================
# 隐藏层反向子页面（结构，见原脚本）
# =========================
def build_back_hidden_layer_page(model: CreatModel,
                                 layer_name: str,
                                 x0: float,
                                 y0: float,
                                 dx: float = 120.0) -> Dict[str, str]:
    elems: List[str] = []

    # 端口：out_pre (in)
    p_out_pre_str, p_out_pre_id, _, _ = model.addPortPlace(x0, y0, 'LR', 'in')      # out_pre (in)
    elems.append(p_out_pre_str)
    # W_next: 来自下一层前向的权重（只读），所以这里设为 'in'
    p_w_next_str, p_w_next_id, _, _ = model.addPortPlace(x0+dx, y0, 'LLR', 'in')    # W_next (in)
    elems.append(p_w_next_str)
    # loss_in (in)
    p_loss_str, p_loss_id, _, _ = model.addPortPlace(x0+2*dx, y0, 'LR', 'in')      # loss_in (in)
    elems.append(p_loss_str)
    # 本层 W_self: I/O（允许回写）
    p_w_str, p_w_id, _, _ = model.addPortPlace(x0+3*dx, y0, 'LLR', 'I/O')          # W_self (I/O)
    elems.append(p_w_str)
    # 本层 B_self: I/O（允许回写）
    p_b_str, p_b_id, _, _ = model.addPortPlace(x0+4*dx, y0, 'LR', 'I/O')           # B_self (I/O)
    elems.append(p_b_str)
    # prev_out (in)
    p_prev_str, p_prev_id, _, _ = model.addPortPlace(x0, y0+60, 'LR', 'in')       # prev_out (in)
    elems.append(p_prev_str)

    # 中间库所：loss_storage、outer
    p_loss_storage_str, p_loss_storage_id, _, _ = model.addPlace(x0+dx, y0+60, 'LR')  # loss_storage
    elems.append(p_loss_storage_str)
    p_outer_str, p_outer_id, _, _ = model.addPlace(x0+3*dx, y0+60, 'LLR')            # outer
    elems.append(p_outer_str)
    p_out_loss_str, p_out_loss_id, _, _ = model.addPortPlace(x0+2*dx, y0+60, 'LR', 'out')  # out_loss (out)
    elems.append(p_out_loss_str)

    # 变迁1：out_pre + W_next + loss_in -> out_loss + loss_storage
    t1_str, t1_id, _, _ = model.addTrance(x0+dx, y0+30)
    elems.append(t1_str)
    elems.append(model.addArc(t1_id, p_out_pre_id, 'PtoT', 'x', x0+0.25*dx, y0))       # out_pre -> t1
    elems.append(model.addArc(t1_id, p_w_next_id, 'PtoT', 'w', x0+0.5*dx, y0))         # W_next (in) -> t1 (read)
    elems.append(model.addArc(t1_id, p_loss_id,    'PtoT', 'l', x0+0.75*dx, y0))       # loss_in -> t1
    elems.append(model.addArc(t1_id, p_out_loss_id,    'TtoP', 'backPropDeltaAuto(w,l,x)', x0+2*dx, y0+60))   # t1 -> out_loss
    elems.append(model.addArc(t1_id, p_loss_storage_id, 'TtoP', '2`backPropDeltaAuto(w,l,x)', x0+dx, y0+60))    # t1 -> loss_storage

    # 变迁2：loss_storage + B_self -> B_self（回写）
    t2_str, t2_id, _, _ = model.addTrance(x0+3.5*dx, y0+30)
    elems.append(t2_str)
    elems.append(model.addArc(t2_id, p_loss_storage_id, 'PtoT', 'l', x0+3*dx, y0+60))   # loss_storage -> t2
    elems.append(model.addArc(t2_id, p_b_id,             'PtoT', 'b', x0+3*dx, y0))       # B_self (I/O) -> t2 (read)
    elems.append(model.addArc(t2_id, p_b_id,             'TtoP', '2`updateBias(b,l,0.2)', x0+3.5*dx, y0))     # t2 -> B_self (I/O) 写回

    # 变迁3：loss_storage + prev_out -> outer（外积）
    t3_str, t3_id, _, _ = model.addTrance(x0+1.5*dx, y0+80)
    elems.append(t3_str)
    elems.append(model.addArc(t3_id, p_loss_storage_id, 'PtoT', 'l', x0+1.25*dx, y0+60)) # loss_storage -> t3
    elems.append(model.addArc(t3_id, p_prev_id,         'PtoT', 'x', x0+1.25*dx, y0+60)) # prev_out -> t3
    elems.append(model.addArc(t3_id, p_outer_id,        'TtoP', 'outerProduct(l,x)', x0+1.75*dx, y0+60)) # t3 -> outer

    # 变迁4：outer + W_self -> W_self（回写）
    t4_str, t4_id, _, _ = model.addTrance(x0+2.5*dx, y0+80)
    elems.append(t4_str)
    elems.append(model.addArc(t4_id, p_outer_id, 'PtoT', 'w1', x0+2.25*dx, y0+60))       # outer -> t4
    elems.append(model.addArc(t4_id, p_w_id,     'PtoT', 'w2', x0+2.25*dx, y0))           # W_self (I/O) -> t4 (read)
    elems.append(model.addArc(t4_id, p_w_id,     'TtoP', '3`updateWeights(w1,w2,0.2)', x0+2.75*dx, y0))           # t4 -> W_self (I/O) 写回

    page_id = model.addPage(elems, [], False)
    return {
        'page_id': page_id,
        'port_out_pre':  p_out_pre_id,
        'port_w_next':   p_w_next_id,   # 现在是 'in'
        'port_loss':     p_loss_id,
        'port_w':        p_w_id,
        'port_b':        p_b_id,
        'port_prev':     p_prev_id,
        'port_out_loss': p_out_loss_id
    }


# =========================
# 主流程（整合前向 + 输出层反向 + 隐藏层反向）
# 增加：主页面上统一的 W/B 读文件源 p_w_src + t_read_w（t_read_w -> 所有 main_W/main_B 的 TtoP）
# =========================
def main(initial: str, structure: str, output: str):
    # 读取模板
    with open(initial, 'r', encoding='utf-8') as f:
        template_lines = [l.rstrip('\n') for l in f]

    # 读取结构
    with open(structure, 'r', encoding='utf-8') as f:
        struct = json.load(f)

    # 提取 Linear 层信息
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

    model.addColor(('list', 'LR', 'REAL'))
    model.addColor(('list', 'LLR', 'LR'))
    model.addColor(('list', 'LI', 'INT'))
    model.addColor("LLRxLR")

    # 读文件配置  与层数有关，每层需要一个权重和偏执变量以绑定库所及函数
    model.addGlobRef("ww0","empty :LLR")
    model.addGlobRef("bb0","empty :LR")
    model.addGlobRef("ww1","empty :LLR")
    model.addGlobRef("bb1","empty :LR")
    model.addGlobRef("ww2","empty :LLR")
    model.addGlobRef("bb2","empty :LR")
    model.addGlobRef("ww3","empty :LLR")
    model.addGlobRef("bb3","empty :LR")
    model.addGlobRef("ww4","empty :LLR")
    model.addGlobRef("bb4","empty :LR")

    model.addMl("""
    fun f_ww0() = (!ww0);
    """)
    model.addMl("""
    fun f_ww1() = (!ww1);
    """)
    model.addMl("""
      fun f_ww2() = (!ww2);""")
    model.addMl("""
    fun f_bb0() = (!bb0);""")
    model.addMl("""
    fun f_bb1() = (!bb1);""")
    model.addMl("""
    fun f_bb2() = (!bb2);""")
    model.addMl("""
       fun f_ww3() = (!ww3);""")
    model.addMl("""
    fun f_bb3() = (!bb3);""")
    model.addMl("""
        fun f_ww4() = (!ww4);""")
    model.addMl("""
    fun f_bb4() = (!bb4);""")


    # 颜色/变量

    model.addGlobRef("m0","empty : ELEM ms")
    model.addGlobRef("m2","empty : ELEM2 ms")
    model.addGlobRef("m3","empty : ELEM3 ms")
    model.addVar("LR", 'x')
    model.addVar("LR", 'b')
    model.addVar("LLR", 'w')
    model.addVar("LI", 'y')
    model.addVar("LR", 'l')
    model.addVar("LLR", "w1")
    model.addVar("LLR", "w2")
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
    model.addMl("""fun elementosIniciais() = (!m0);
              """)
    model.addMl("""fun elementosIniciais2() = (!m2);
              """)
    model.addMl("""fun elementosIniciais3() = (!m3);
          """)
    model.addMl("""fun softmax (xs: real list) : real list =
        let
          val exps = List.map Math.exp xs
          val sumExps = List.foldl op+ 0.0 exps
        in
          List.map (fn x => x / sumExps) exps
        end;
          """)
    model.addMl("""
      fun outerProduct ([], _) = []
    | outerProduct (x::xs, v: real list) : real list list =
        (List.map (fn y => x * y) v) :: outerProduct(xs, v);
      """)
    model.addMl("""
      fun subIntRealList (ints: int list, reals: real list) : real list =
      let
          (*  *)
          fun subElem (i, r) = Real.fromInt(i) - r
      in
          ListPair.map subElem (ints, reals)
      end;
      """)
    model.addMl("""
      fun updateBias ([], [], _) : real list = []
    | updateBias (b::bs, g::gs, lr: real) : real list =
        (b - lr * g) :: updateBias(bs, gs, lr)
    | updateBias (_, _, _) = raise Fail "Lists must have the same length";
      """)
    model.addMl("""
      fun updateWeights ([], [], _) : real list list = []
    | updateWeights (wRow::wRows, gRow::gRows, lr: real) : real list list =
        let
            (*  *)
            fun updateRow ([], [], _) : real list = []
              | updateRow (w::ws, g::gs, lr) = (w - lr * g) :: updateRow(ws, gs, lr)
              | updateRow (_, _, _) = raise Fail "Row lengths must match"
        in
            updateRow(wRow, gRow, lr) :: updateWeights(wRows, gRows, lr)
        end
    | updateWeights (_, _, _) = raise Fail "Matrix sizes must match";
      """)
    model.addMl("""
      fun backPropError (weights, delta) =
      let
          val nCols = length (hd weights)
          (*  *  *)
          fun calcCol colIdx =
              List.foldl (fn (row, acc) => acc + List.nth(row, colIdx) * List.nth(delta, 0)) 0.0 weights
          fun helper i acc =
              if i >= nCols then rev acc
              else helper (i+1) (calcCol(i)::acc)
      in
          helper 0 []
      end;
      """)
    model.addMl("""
     fun backPropDeltaAuto (weights: LLR, deltaNext: LR, z: LR) : LR =
      let
       val nCols = length z (* *)
        val nRows = length deltaNext (* *)

         (* deltaNext *) 
         fun linearDelta(colIdx:int) : REAL = 
         let
          fun sumRow(rows: LLR, deltas: LR, idx:int) : REAL =
           case (rows, deltas) of
            ([], []) => 0.0
             | (r::rs, d::ds) => List.nth(r, idx) * d + sumRow(rs, ds, idx)
              | _ => raise Fail "Matrix sizes mismatch"
               in
                sumRow(weights, deltaNext, colIdx)
                 end
                  (* (l) = (_next * W) f'(z) *) 
                  fun computeDelta(i:int, acc: LR) : LR =
                   if i >= nCols then rev acc 
                   else
                    let
                     val linear = linearDelta(i)
                      val grad = if List.nth(z, i) > 0.0 then linear else 0.0 (* ReLU *)
                       in
                        computeDelta(i+1, grad::acc)
                         end
                          in
                           computeDelta(0, [])
                            end;
      """)

    # ====== 子页面（前向） ======
    page_fwd_info: List[Dict[str, str]] = []
    x0_base = 120.0
    y0_base = 150.0
    dx = 120.0
    horiz_gap = 6*dx + 120

    for idx, layer in enumerate(mlp_layers_src):
        name = layer.get('name', f'layer{idx}')
        in_f  = layer.get('in_features', 1)
        out_f = layer.get('out_features', 1)
        x0 = x0_base + idx*horiz_gap
        is_last = (idx == len(mlp_layers_src)-1)
        info = build_fwd_layer_page_as_ports(model, name, in_f, out_f, x0, y0_base, is_last, dx=dx)
        info['name'] = name
        page_fwd_info.append(info)

    # ====== 子页面（反向：输出层） ======
    x_back0 = x0_base + (len(mlp_layers_src))*horiz_gap
    back_last_info = build_back_last_layer_page(model, "back_last", x_back0, y0_base, dx=dx)

    # ====== 主页面构建 ======
    elems: List[str] = []
    subtrans: List[str] = []

    # ====== 新增：文件读取源与读取变迁（x） ======
    p_x_src_str, p_x_src_id, _, _ = model.addPlace_init(20, 360, 'E',initmark='c')  # 文件源 -> x
    elems.append(p_x_src_str)
    t_read_x_str, t_read_x_id, _, _ = model.addTrance_file(50, 360,"""action
let
  val arq_ent = TextIO.openIn(
  "C:/Users/CC/Desktop/output/x_test.txt"
  );
  val dados = ELEM.input_ms(arq_ent);
in
  m0 := dados;
  TextIO.closeIn(arq_ent);
  ()
end;""")
    elems.append(t_read_x_str)

    # ====== 新增：文件读取源与读取变迁（y） ======
    p_y_src_str, p_y_src_id, _, _ = model.addPlace_init(20, 260, 'E',initmark='c')  # 文件源 -> y
    elems.append(p_y_src_str)
    t_read_y_str, t_read_y_id, _, _ = model.addTrance_file(50, 260,"""action
let
  val arq_ent = TextIO.openIn(
  "C:/Users/CC/Desktop/output/y_test.txt"
  );
  val dados = ELEM3.input_ms(arq_ent);
in
  m3 := dados;
  TextIO.closeIn(arq_ent);
  ()
end;
    """)
    elems.append(t_read_y_str)

    # ====== 新增：统一的 W/B 文件读取源与读取变迁（p_w_src / t_read_w） ======
    # 这是新增的主页面库所+变迁：t_read_w 的多个出弧会指向后面创建的所有 main_W / main_B（见下面 forward loop 中的注释）
    p_w_src_str, p_w_src_id, _, _ = model.addPlace_init(20, 200, 'E', initmark='c')  # ← 新增：权重/偏置文件源库所（主页面）
    elems.append(p_w_src_str)

    #读权重与变迁所绑定的函数，与上文中的ww bb所对应
    t_read_w_str, t_read_w_id, _, _ = model.addTrance_file(50, 200,"""
    action
let
  fun load_layer (wfile: string, bfile: string) : LLRxLR =
    let
      val wf = TextIO.openIn(wfile);
      val bf = TextIO.openIn(bfile);
      val w = LLR.input(wf);
      val b = LR.input(bf);
    in
      TextIO.closeIn(wf);
      TextIO.closeIn(bf);
      (w, b)
    end;

  (*  *)
  val l0 = load_layer("C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__0__weight__init.txt",
                      "C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__0__bias__init.txt");
  val l1 = load_layer("C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__2__weight__init.txt",
                      "C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__2__bias__init.txt");
  val l2 = load_layer("C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__4__weight__init.txt",
                      "C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__4__bias__init.txt");
  val l3 = load_layer("C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__6__weight__init.txt",
                      "C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__6__bias__init.txt");
  val l4 = load_layer("C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__8__weight__init.txt",
                      "C:/Users/Admin/Desktop/pycode-to-cpn/maintool/mlp2_export/net__8__bias__init.txt");                                        

  val (w_0, b_0) = l0;
  val (w_1, b_1) = l1;
  val (w_2, b_2) = l2;
  val (w_3, b_3) = l3;
  val (w_4, b_4) = l4;

in
  (*  *)
  ww0 := w_0;  bb0 := b_0;
  ww1 := w_1;  bb1 := b_1;
  ww2 := w_2;  bb2 := b_2;
  ww3 := w_3;  bb3 := b_3;
  ww4 := w_4;  bb4 := b_4;


 
()
end;
    """)  # ← 新增：读取所有 W/B 的统一变迁（主页面）
    elems.append(t_read_w_str)
    # 弧（p_w_src -> t_read_w）：把文件源与读取变迁相连（PtoT）
    elems.append(model.addArc(t_read_w_id, p_w_src_id, 'PtoT', 'c', 40, 200))  # ← 新增：读W/B 的输入弧

    # ====== 主页面：X, y（由读取变迁写入） ======
    p_x_str, p_x_id, _, _ = model.addPlace(80, 360, 'LR')  # p_x（前向用）
    elems.append(p_x_str)
    p_y_str, p_y_id, _, _ = model.addPlace(80, 260, 'LI')  # p_y（标签）
    elems.append(p_y_str)

    # ====== x 的快照库所：p_x_mid（供反向页用） ======
    p_x_mid_str, p_x_mid_id, _, _ = model.addPlace(140, 360, 'LR')  # p_x 的快照
    elems.append(p_x_mid_str)

    # 弧：p_x_src -> t_read_x (PtoT)；t_read_x -> p_x (TtoP)；t_read_x -> p_x_mid (TtoP)
    elems.append(model.addArc(t_read_x_id, p_x_src_id, 'PtoT', 'c', 40, 360))  # 读 x 源
    elems.append(model.addArc(t_read_x_id, p_x_id, 'TtoP', 'elementosIniciais()', 80, 360))      # 写 p_x
    elems.append(model.addArc(t_read_x_id, p_x_mid_id, 'TtoP', 'elementosIniciais()', 120, 360))  # 直接写快照（新增的第二个出弧）

    # 弧：p_y_src -> t_read_y (PtoT)；t_read_y -> p_y (TtoP)
    elems.append(model.addArc(t_read_y_id, p_y_src_id, 'PtoT', 'c', 40, 260))
    elems.append(model.addArc(t_read_y_id, p_y_id, 'TtoP', 'elementosIniciais3()', 80, 260))

    # logits / softmax / loss
    p_logits_str, p_logits_id, _, _ = model.addPlace(80, 460, 'LR')  # logits
    elems.append(p_logits_str)
    p_softmax_str, p_softmax_id, _, _ = model.addPlace(220, 460, 'LR')  # softmax
    elems.append(p_softmax_str)
    p_loss_str, p_loss_id, _, _ = model.addPlace(360, 460, 'LR')  # loss
    elems.append(p_loss_str)

    # 每层主页面上的 W / B / OUT / OUT_SNP / OUT_PRE
    main_W: List[str] = []
    main_B: List[str] = []
    main_OUT: List[str] = []
    main_OUT_SNP: List[str] = []
    main_OUT_PRE: List[str] = []

    prev_out_place = p_x_id  # 前向第一层输入来自 p_x

    # 前向层实例化（主页 places 与子页面端口映射）
    for idx, fwd in enumerate(page_fwd_info):
        px = 200 + idx*300
        pw_str, pw_id, _, _ = model.addPlace(px, 140, 'LLR'); elems.append(pw_str); main_W.append(pw_id)  # main W
        pb_str, pb_id, _, _ = model.addPlace(px, 200, 'LR');  elems.append(pb_str); main_B.append(pb_id)   # main B
        p_out_str, p_out_id, _, _ = model.addPlace(px, 360, 'LR'); elems.append(p_out_str); main_OUT.append(p_out_id)  # main OUT

        # ====== 新增：让 t_read_w 把读取到的数据写入各个 main_W / main_B
        # 说明：这里我们把统一读取变迁 t_read_w 的多个出弧连接到每个主页面权重与偏置库所（TtoP）。
        # 你可以把 'elementosIniciais2()' / 'elementosIniciais()' 替换为你自己的写回函数。
        elems.append(model.addArc(t_read_w_id, pw_id, 'TtoP', f"f_ww{idx}()", px+10, 160))  # ← 新增：t_read_w -> main weight place
        elems.append(model.addArc(t_read_w_id, pb_id, 'TtoP', f'f_bb{idx}()', px+10, 200))   # ← 新增：t_read_w -> main bias place

        out_snap_id = None
        out_preact_id = None
        if 'port_out_snap' in fwd:
            p_snap_str, p_snap_id, _, _ = model.addPlace(px, 400, 'LR'); elems.append(p_snap_str); out_snap_id = p_snap_id; main_OUT_SNP.append(p_snap_id)
        else:
            main_OUT_SNP.append("")
        if 'port_out_pre' in fwd:
            p_pre_str, p_pre_id, _, _ = model.addPlace(px, 440, 'LR'); elems.append(p_pre_str); out_preact_id = p_pre_id; main_OUT_PRE.append(p_pre_id)
        else:
            main_OUT_PRE.append("")

        # portsock：把子页的 port_w/port_b（I/O）都映射到 main_W/main_B
        portsock = []
        portsock.append(f"({fwd['port_in']},{prev_out_place})")
        portsock.append(f"({fwd['port_w']},{pw_id})")
        portsock.append(f"({fwd['port_b']},{pb_id})")
        portsock.append(f"({fwd['port_out']},{p_out_id})")
        if out_snap_id:
            portsock.append(f"({fwd['port_out_snap']},{out_snap_id})")
        if out_preact_id:
            portsock.append(f"({fwd['port_out_pre']},{out_preact_id})")
        portsock_str = "".join(portsock)

        t_sub_str, t_sub_id, _, _ = model.addSubTrance(px, 300, fwd['page_id'], portsock_str, fwd['name'])
        elems.append(t_sub_str)

        # 弧：方向符合端口类型（子页 W/B 为子页 I/O，主页面既有 PtoT，又有 TtoP）
        elems.append(model.addArc(t_sub_id, prev_out_place, 'PtoT', 'x', px-40, 330))
        elems.append(model.addArc(t_sub_id, pw_id, 'PtoT', 'w', px-40, 160))
        elems.append(model.addArc(t_sub_id, pw_id, 'TtoP', 'w', px+40, 160))
        elems.append(model.addArc(t_sub_id, pb_id, 'PtoT', 'b', px-40, 200))
        elems.append(model.addArc(t_sub_id, pb_id, 'TtoP', 'b', px+40, 200))
        elems.append(model.addArc(t_sub_id, p_out_id, 'TtoP', 'x', px+40, 330))
        if out_snap_id:
            elems.append(model.addArc(t_sub_id, out_snap_id, 'TtoP', 'x', px+40, 370))
        if out_preact_id:
            elems.append(model.addArc(t_sub_id, out_preact_id, 'TtoP', 'x', px+40, 410))

        subtrans.append(t_sub_id)
        prev_out_place = p_out_id

    # softmax & loss
    last_out = main_OUT[-1]
    t_soft_str, t_soft_id, _, _ = model.addTrance(220, 420)
    elems.append(t_soft_str)
    elems.append(model.addArc(t_soft_id, last_out,     'PtoT', 'x', 200, 380))
    elems.append(model.addArc(t_soft_id, p_softmax_id, 'TtoP', 'softmax(x)', 240, 460))

    t_loss_str, t_loss_id, _, _ = model.addTrance(360, 420)
    elems.append(t_loss_str)
    elems.append(model.addArc(t_loss_id, p_softmax_id, 'PtoT', 'x', 320, 440))
    elems.append(model.addArc(t_loss_id, p_y_id,       'PtoT', 'y', 320, 280))
    elems.append(model.addArc(t_loss_id, p_loss_id,    'TtoP', '2`subIntRealList(y,x)', 400, 460))

    # ====== back_last (output layer) 替代变迁，以及 main_back_out place (接误差) ======
    portsock_back = []
    portsock_back.append(f"({back_last_info['port_loss']},{p_loss_id})")
    portsock_back.append(f"({back_last_info['port_w']},{main_W[-1]})")   # 子页 W 为 I/O
    portsock_back.append(f"({back_last_info['port_b']},{main_B[-1]})")   # 子页 B 为 I/O

    # 关键：prev_snapshot 的兜底为 p_x_mid_id（使用快照而不是直接 p_x）
    prev_snapshot = main_OUT_SNP[-2] if len(main_OUT_SNP) >= 2 and main_OUT_SNP[-2] else p_x_mid_id
    portsock_back.append(f"({back_last_info['port_prev']},{prev_snapshot})")

    mb_out_px = 680
    mb_out_py = 420
    mb_out_str, mb_out_id, _, _ = model.addPlace(mb_out_px+40, mb_out_py, 'LR')  # main_back_out for back_last
    elems.append(mb_out_str)
    portsock_back.append(f"({back_last_info['port_out_loss']},{mb_out_id})")
    portsock_back_str = "".join(portsock_back)

    t_back_str, t_back_id, _, _ = model.addSubTrance(520, 420, back_last_info['page_id'],
                                                     portsock_back_str, "back_last")
    elems.append(t_back_str)
    elems.append(model.addArc(t_back_id, p_loss_id, 'PtoT', 'd', 500, 460))
    elems.append(model.addArc(t_back_id, main_W[-1], 'PtoT', 'w', 500, 160))
    elems.append(model.addArc(t_back_id, main_W[-1], 'TtoP', 'w', 540, 160))
    elems.append(model.addArc(t_back_id, main_B[-1], 'PtoT', 'b', 500, 200))
    elems.append(model.addArc(t_back_id, main_B[-1], 'TtoP', 'b', 540, 200))
    elems.append(model.addArc(t_back_id, prev_snapshot, 'PtoT', 'x', 500, 400))
    elems.append(model.addArc(t_back_id, mb_out_id, 'TtoP', 'g', 560, 420))
    subtrans.append(t_back_id)

    main_back_outs: List[str] = [mb_out_id]

    # ====== 反向：隐藏层逐层构建（从倒数第二层向前） ======
    for idx in range(len(page_fwd_info)-2, -1, -1):
        layer_name = page_fwd_info[idx]['name']
        x_back = x_back0 + (len(page_fwd_info)-2 - idx) * horiz_gap
        hidden_back_info = build_back_hidden_layer_page(model, f"back_{layer_name}", x_back, y0_base, dx=dx)

        portsock = []
        if main_OUT_PRE[idx]:
            portsock.append(f"({hidden_back_info['port_out_pre']},{main_OUT_PRE[idx]})")
        else:
            portsock.append(f"({hidden_back_info['port_out_pre']},{main_OUT[idx]})")
        # 注意：hidden_back_info['port_w_next'] 在这里是 'in'（只读）
        portsock.append(f"({hidden_back_info['port_w_next']},{main_W[idx+1]})")

        if idx == len(page_fwd_info)-2:
            portsock.append(f"({hidden_back_info['port_loss']},{p_loss_id})")
            loss_source_place = p_loss_id
        else:
            loss_source_place = main_back_outs[-1]
            portsock.append(f"({hidden_back_info['port_loss']},{loss_source_place})")

        portsock.append(f"({hidden_back_info['port_w']},{main_W[idx]})")   # 本层 W (子页 I/O)
        portsock.append(f"({hidden_back_info['port_b']},{main_B[idx]})")   # 本层 B (子页 I/O)

        # 最前层 prev_snap 使用 p_x_mid_id（快照），或上一层 out_snap（若存在）
        if idx == 0:
            prev_snap_for_portsock = p_x_mid_id
        else:
            prev_snap_for_portsock = main_OUT_SNP[idx-1] if idx-1 >= 0 and main_OUT_SNP[idx-1] else p_x_mid_id
        portsock.append(f"({hidden_back_info['port_prev']},{prev_snap_for_portsock})")

        create_main_out_for_this_layer = (idx != 0)
        if create_main_out_for_this_layer:
            mb_px = 680
            mb_py = 420 + (len(page_fwd_info)-2 - idx) * 60
            mb_str, mb_id, _, _ = model.addPlace(mb_px+40, mb_py, 'LR')
            elems.append(mb_str)
            portsock.append(f"({hidden_back_info['port_out_loss']},{mb_id})")
        portsock_str = "".join(portsock)

        sub_x = 520
        sub_y = 420 + (len(page_fwd_info)-2 - idx) * 60
        t_str, t_id, _, _ = model.addSubTrance(sub_x, sub_y, hidden_back_info['page_id'],
                                               portsock_str, f"back_{layer_name}")
        elems.append(t_str)

        # 添加弧：方向与端口类型一致
        elems.append(model.addArc(t_id, loss_source_place, 'PtoT', 'd', sub_x-20, sub_y+20))
        bind_out_pre = main_OUT_PRE[idx] if main_OUT_PRE[idx] else main_OUT[idx]
        elems.append(model.addArc(t_id, bind_out_pre, 'PtoT', 'x', sub_x-20, sub_y-20))
        # W_next 是 port 'in'，主页面使用 PtoT 连接 main_W[idx+1] -> sub
        elems.append(model.addArc(t_id, main_W[idx+1], 'PtoT', 'w', sub_x-20, sub_y-80))
        # 本层 W/B 为子页 I/O：主页面同时做 PtoT 与 TtoP
        elems.append(model.addArc(t_id, main_W[idx], 'PtoT', 'w', sub_x-20, sub_y-40))
        elems.append(model.addArc(t_id, main_W[idx], 'TtoP', 'w', sub_x+20, sub_y-40))
        elems.append(model.addArc(t_id, main_B[idx], 'PtoT', 'b', sub_x-20, sub_y-10))
        elems.append(model.addArc(t_id, main_B[idx], 'TtoP', 'b', sub_x+20, sub_y-10))
        elems.append(model.addArc(t_id, prev_snap_for_portsock, 'PtoT', 'x', sub_x-20, sub_y+40))

        if create_main_out_for_this_layer:
            elems.append(model.addArc(t_id, mb_id, 'TtoP', 'g', sub_x+40, sub_y+20))
            main_back_outs.append(mb_id)

        subtrans.append(t_id)

    # 写主页面
    model.addPage(elems, [subtrans], True)

    # 输出到文件
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write("\n".join(model.getCpntxt()))
    print("Generated:", output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build CPN of MLP (forward + multi-layer backprop pages, W/B I/O in subpages, unified W/B file-read transition)")
    parser.add_argument('--initial', required=True, help='initial template cpn file (initial.cpn)')
    parser.add_argument('--structure', required=True, help='mlp structure json (mlp_structure.json)')
    parser.add_argument('--output', default='output/mlp_train_with_x_snapshot_and_w_read.cpn', help='output cpn file')
    args = parser.parse_args()
    main(args.initial, args.structure, args.output)
