import os
import copy

def Get_flops(feature_size, model_type, stride_type):

    feature_size_str = "-".join(list(map(str, feature_size)))
    model_type_str = "-".join(list(map(str, model_type)))
    stride_type_str = "-".join(list(map(str, stride_type)))

    cmd = "python ./init_model.py --fm=" + feature_size_str + " --mt=" \
          + model_type_str + " --st=" + stride_type_str

    r = os.popen(cmd)
    for line in r.readlines():
        if 'total' in line:
            line_new = " ".join(line.split())
            return float(line_new.split(" ")[3].replace(",",""))


def read_liner_model(name):
    path = './liner_model/'+str(name)+".txt"
    with open(path, 'r') as fr:
        data = fr.readlines()[0].strip().split("-")
    return float(data[0])*1E-10, float(data[1])


def Predicted_time(device_type, feature_size, model_type, stride_type, bandwidth):
    total_flops = Get_flops(feature_size, model_type, stride_type)

    runing_time_1, runing_time_2 = read_liner_model(device_type)
    run_time = runing_time_1*total_flops + runing_time_2

    trains_upload_time = (feature_size[0] * (feature_size[1]-4) * 32 * model_type[0])/(bandwidth * 1024 * 1024)
    trains_download_time = ((feature_size[0]-4) * (feature_size[1]-4) * 32 * model_type[-1]) / (bandwidth * 1024 * 1024)

    return run_time+trains_upload_time+trains_download_time


def Optimal_One_Dimensional_Partition(device, feature_size, model_type, stride_type, bandwidth):
    w = [0 for _ in range(len(device))] #map
    ws = 1
    dc = [] #算力
    dt = [0 for _ in range(len(device))] #时间
    gp = 100000
    w_old = []
    for d in device:
        if d == 'Pi3B+':
            dc.append(1)
        elif d == 'Pi4B':
            dc.append(1.5)

    count = 1
    while True:
        if count == 1:
            sl = int(feature_size[0] / sum(dc))
            for i in range(len(w)-1):
                w[i] = int(dc[i] * sl)
            w[-1] = int(feature_size[0] - sum(w))
            w_old = w
        else:
            for i in range(len(w)):
                dt[i] = Predicted_time(device_type= device[i],
                                       feature_size=[w[i]+4, feature_size[1]+4],
                                       model_type=model_type,
                                       stride_type=stride_type,
                                       bandwidth=bandwidth[i])

            max_index = dt.index(max(dt))
            min_index = dt.index(min(dt))

            gp_temp = dt[max_index] - dt[min_index]

            if count > 1 and gp_temp >= gp or w[max_index] == ws or gp_temp == 0:
                break

            w_old = copy.deepcopy(w)

            gp = gp_temp

            for i in range(len(w)):
                if i == max_index:
                    w[i] = w[i] - ws
                elif i == min_index:
                    w[i] = w[i] + ws
                else:
                    w[i] = w[i]
        count += 1
    return w_old


if __name__ == '__main__':
    w = Optimal_One_Dimensional_Partition(device = ['Pi3B+', 'Pi3B+'],
                   feature_size = [12, 12],
                   model_type = [256, 512, 512],
                   stride_type = [1, 1],
                   bandwidth = [160, 160])

    print("各设备划分方案:{}".format(w))