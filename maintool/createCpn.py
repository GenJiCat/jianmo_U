import sys
from createModel import CreatModel


def add_cpntest(creat_model: CreatModel, cpntxt: list):
    # 初始化模型文本
    creat_model.setCpntxt(cpntxt)
    # 准备变量
    news = []
    port_place = []
    socket_place = []
    fusion_place = []
    sub_trans = []
    trans = []
    fu_no = 1
    fusion_name = f"fusion{fu_no}"

    # 全局色彩、变量、ML函数
    creat_model.addColor(('list','LR','REAL'))
    creat_model.addColor(('list','LLR','LR'))
    # creat_model.addGlobRef("m0","empty : ELEM2 ms")
    creat_model.addVar("LR",'x')
    creat_model.addVar("LR",'b')
    creat_model.addVar("LLR",'w')
    creat_model.addMl("""fun    mult([],v,answer_vector) = rev(answer_vector)
     |   mult(m,v,answer_vector)=
let
      val line = hd m

fun     mlc([],[],result)=result
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
    creat_model.addColor("INTxSTRING")
    creat_model.addVar("INT", "n")
    creat_model.addVar("STRING", "p")
    creat_model.addMl(
        "val AllPackets =\n"
        "1`(1,&quot;COL&quot; )++\n"
        "1`(2,&quot;OUR&quot;)++\n"
        "1`(3,&quot;ED &quot;)++\n"
        "1`(4,&quot;PET&quot;)++\n"
        "1`(5,&quot;RI  &quot;)++\n"
        "1`(6,&quot;NET&quot;);")
    creat_model.addMl(
        "fun fun2(n,p)=\n"
        "let \n"
        "   val n=n+3 \n"
        "in (n,p)\n"
        "end;")

    # 起始坐标
    x, y = -420.0, 168.0

    # 子页面端口库所 In
    newP = creat_model.addPortPlace(x, y, "INTxSTRING", "In")
    news.append(newP[0])
    port_place.append(newP[1])

    # 新变迁
    x += 210.0
    newT = creat_model.addTrance(x, y, True)
    news.append(newT[0])

    # 弧 P->T
    ax = (float(newP[2]) + float(newT[2])) / 2
    ay = float(newP[3]) + 11
    newA = creat_model.addArc(newT[1], newP[1], "PtoT", "(n,p)", ax, ay)
    news.append(newA)

    # 子页面端口库所 Out
    x += 210.0
    newP = creat_model.addPortPlace(x, y, "INTxSTRING", "Out")
    news.append(newP[0])
    port_place.append(newP[1])
    ax = (float(newP[2]) + float(newT[2])) / 2
    newA = creat_model.addArc(newT[1], newP[1], "TtoP", "fun2(n,p)", ax, ay)
    news.append(newA)

    # 普通带 initmark 的库所、变迁、融合集库所 示例
    for detail_y, black in [(-42.0, False)]:
        x, y = -420.0, detail_y
        newP = creat_model.addPlace_init(x, y, "INTxSTRING", initmark='1`c', black=black)
        news.append(newP[0])
        x += 210.0
        newT = creat_model.addTrance(x, y, False)
        news.append(newT[0])
        ax = (float(newP[2]) + float(newT[2])) / 2
        ay = float(newP[3]) + 11
        newA = creat_model.addArc(newT[1], newP[1], "PtoT", "(n,p)", ax, ay)
        news.append(newA)
        x += 210.0
        newP = creat_model.addFusionPlace(x, y, "INTxSTRING", fusion_name)
        news.append(newP[0])
        fusion_place.append(newP[1])
        ax = (float(newP[2]) + float(newT[2])) / 2
        newA = creat_model.addArc(newT[1], newP[1], "TtoP", "fun2(n,p)", ax, ay)
        news.append(newA)

    # 创建子页面
    subpage = creat_model.addPage(news, sub_trans, False)

    # 主页面槽库所
    news = []
    x, y = -420.0, 168.0
    for dx in [0, 420.0]:
        newP = creat_model.addPlace(x + dx, y, "INTxSTRING", black=False)
        news.append(newP[0])
        socket_place.append(newP)

    # 替代变迁 portsock 拼接
    portsock = "".join(f"({pp},{sp[1]})" for pp, sp in zip(port_place, socket_place))
    x -= 210.0
    newT = creat_model.addSubTrance(x, y, subpage, portsock, "sub")
    news.append(newT[0])
    trans.append(newT[1])

    # 弧 from socket place to sub trance
    for idx, orient, text in [(0, "PtoT", "(n,p)"), (1, "TtoP", "fun2(n,p)")]:
        sp = socket_place[idx]
        ax = (float(sp[2]) + float(newT[2])) / 2
        ay = float(sp[3]) + 11
        newA = creat_model.addArc(newT[1], sp[1], orient, text, ax, ay)
        news.append(newA)

    # 其他普通与融合集例子
    for detail_y in [-42.0, -242.0]:
        x, y = -420.0, detail_y
        newP = creat_model.addPlace(x, y, "INTxSTRING", black=False)
        news.append(newP[0])
        x += 210.0
        newT = creat_model.addTrance(x, y, False)
        news.append(newT[0])
        ax = (float(newP[2]) + float(newT[2])) / 2
        ay = float(newP[3]) + 11
        newA = creat_model.addArc(newT[1], newP[1], "PtoT", "(n,p)", ax, ay)
        news.append(newA)
        x += 210.0
        newP = creat_model.addFusionPlace(x, y, "INTxSTRING", fusion_name)
        news.append(newP[0])
        fusion_place.append(newP[1])
        ax = (float(newP[2]) + float(newT[2])) / 2
        newA = creat_model.addArc(newT[1], newP[1], "TtoP", "fun2(n,p)", ax, ay)
        news.append(newA)

    # 添加最终 fusion
    # creat_model.addFusion(fusion_place)

    # 嵌套实例页面
    sub_trans.append(trans)
    print(news)
    creat_model.addPage(news, sub_trans, True)


if __name__ == "__main__":

    input_file = "initial.cpn"
    output_file = "../output66.cpn"
    # 读取原始 cpn 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    model = CreatModel()
    add_cpntest(model, lines)
    # 输出修改后的模型
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(model.getCpntxt()))
    print(f"Generated new CPN model to {output_file}")
