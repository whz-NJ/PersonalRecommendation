#看看文件内容有多少列
if __name__ ==  "__main__":
    fp = open("../data/lr_coef")
    count = 0
    for line in fp:
        item = line.strip().split(",")
        print (len(item))

