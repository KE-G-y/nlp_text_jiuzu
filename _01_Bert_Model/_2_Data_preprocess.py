# 2.读取数据，并将 text \t label  转变成 [ ( text, label) ] 格式
def load_raw_data(datapath):
    data =  []
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            text, label = line.split("\t")
            data.append((text, int(label)))
    return data