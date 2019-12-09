import numpy as np
import pandas as pd
from collections import Counter
import time

allow_duplicate = False
def load_data(csv_path):
    data = pd.read_csv(csv_path,sep=";",dtype=np.float64)
    # data = data.sample(frac=1)
    # data = data.reset_index(drop=True)
    label = data["quality"]
    data = data.drop(["quality"], axis=1)
    return data.values,label,data.columns.values

class KDNode():
    def __init__(self,value,label,left,right,depth):
        self.value = value
        self.label = label
        self.left = left
        self.right = right
        self.depth = depth

class KDTree():
    def __init__(self,values,labels):
        self.values = values
        self.labels = labels
        if(len(self.values) == 0 ):
            raise Exception('Data For KD-Tree Must Be Not empty.')
        self.dims_len = len(self.values[0])
        self.root = self.build_KDTree()
        self.KNN_result = []
        self.nums=0

    def build_KDTree(self):
        data = np.column_stack((self.values,self.labels))
        return self.build_KDTree_core(data,0)

    def dist(self,point1,point2):
        return np.sqrt(np.sum((point1-point2)**2))

    #data:带标签的数据且已经排好序的
    def build_KDTree_core(self,data,depth):
        if len(data)==0:
            return None
        cuttint_dim = depth % self.dims_len

        data = data[data[:, cuttint_dim].argsort()]  # 按照第当前维度排序
        mid_index = len(data)//2
        node = KDNode(data[mid_index,:-1],data[mid_index,-1],None,None,depth)
        node.left = self.build_KDTree_core(data[0:mid_index],depth+1)
        node.right = self.build_KDTree_core(data[mid_index+1:], depth + 1)
        return node

    def search_KNN(self,target,K):
        if self.root is None:
            raise Exception('KD-Tree Must Be Not empty.')
        if K > len(self.values):
            raise ValueError("K in KNN Must Be Greater Than Lenght of data")
        if len(target) !=len(self.root.value):
            raise ValueError("Target Must Has Same Dimension With Data")
        self.KNN_result = []
        self.nums = 0
        self.search_KNN_core(self.root,target,K)
        return self.nums


    def search_KNN_core(self,root, target, K):
        if root is None:
            return
        cur_data = root.value
        label = root.label
        self.nums+=1
        distance = self.dist(cur_data,target)

        is_duplicate = [kd_tree.dist(cur_data, item[0].value)< 1e-4 and
                        abs(label-item[0].label) < 1e-4 for item in self.KNN_result]
        if not np.array(is_duplicate, np.bool).any() or allow_duplicate:
            if len(self.KNN_result) < K:
                # 向结果中插入新元素
                self.KNN_result.append((root,distance))
            elif distance < self.KNN_result[0][1]:
                # 替换结果中距离最大元素
                self.KNN_result = self.KNN_result[1:]+[(root,distance)]
        self.KNN_result=sorted(self.KNN_result,key=lambda x:-x[1])
        cuttint_dim = root.depth % self.dims_len
        if abs(target[cuttint_dim] - cur_data[cuttint_dim]) < self.KNN_result[0][1] or len(self.KNN_result) < K:
            # 在当前切分维度上,以target为中心,最近距离为半径的超体小球如果和该维度上的超平面有交集,那么说明可能还存在更近的数据点
            # 同时如果还没找满K个点，也要继续寻找(这种有选择的比较,正是使用KD树进行KNN的优化之处,不用像一般KNN一样在整个数据集遍历)
            self.search_KNN_core(root.left,target,K)
            self.search_KNN_core(root.right,target,K)
        # 在当前划分维度上,数据点小于超平面,那么久在左子树继续找,否则在右子树继续找
        elif target[cuttint_dim] < cur_data[cuttint_dim]:
            self.search_KNN_core(root.left,target,K)
        else:
            self.search_KNN_core(root.right,target,K)


if __name__ == '__main__':

    csv_path = "winequality-white.csv"
    data,lables,dim_label = load_data(csv_path)
    split_rate = 0.8;K=5
    train_num = int(len(data)*split_rate)
    print(len(data),train_num)

    start1 = time.time()
    kd_tree = KDTree(data[:train_num],lables[:train_num])
    end1 = time.time()

    diff_all = 0
    accuracy = 0
    search_all_time = 0
    calu_dist_nums=0
    for index,target in enumerate(data[train_num:]):
        start2 = time.time()
        calu_dist_nums+=kd_tree.search_KNN(target, K)
        end2 = time.time()
        search_all_time+=end2-start2

        keys = [tuple(node[0].value) for node in kd_tree.KNN_result]
        # for res in kd_tree.KNN_result:
        #     print("res:",res[0].value,res[0].label,res[1])
        pred_label = Counter(node[0].label for node in kd_tree.KNN_result).most_common(1)[0][0]
        diff_all += abs(lables[index]-pred_label)
        if (lables[index] - pred_label) <= 1e-5:
            accuracy += 1
        print("accuracy:", accuracy / (index + 1))
        print("Total:{},MSE:{:.3f}    {}--->{}".format(index + 1, (diff_all / (index + 1)), lables[index],
                                                        pred_label))
        # print("----")

    print("KDtree构建时间：", end1 - start1)
    print("程序运行时间：", search_all_time/len(data[train_num:]))
    print("平均计算次数：", calu_dist_nums / len(data[train_num:]))

        # 暴力KNN验证
        # KNN_res = []
        # for index2, curData in enumerate(data[:train_num]):
        #     is_duplicate = [kd_tree.dist(curData, v[0]) < 1e-4 for v in KNN_res]
        #     if np.array(is_duplicate, np.bool).any() and not allow_duplicate:
        #         continue
        #     cur_dist = kd_tree.dist(curData, target)
        #     if len(KNN_res) < K:
        #         KNN_res.append((curData, lables[index2], cur_dist))
        #     elif cur_dist < KNN_res[0][2]:
        #         KNN_res = KNN_res[1:] + [(curData, lables[index2], cur_dist)]
        #     KNN_res = sorted(KNN_res, key=lambda x: -x[2])
        # pred_label2 = Counter(node[1] for node in KNN_res).most_common(1)[0][0]
        # for my_res in KNN_res:
        #     print("res:", my_res[0], my_res[1], my_res[2])
        # print("--------------{}--->{} vs {}------------------".format(lables[index], pred_label, pred_label2))



