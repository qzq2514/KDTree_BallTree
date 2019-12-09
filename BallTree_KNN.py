import numpy as np
import pandas as pd
from collections import Counter
import time

allow_duplicate = False
def load_data(csv_path):
    data = pd.read_csv(csv_path,sep=";")
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)
    label = data["quality"]
    data = data.drop(["quality"], axis=1)
    return data.values,label,data.columns.values

class Ball():
    def __init__(self,center,radius,points,left,right):
        self.center = center      #使用该点即为球中心,而不去精确地去找最小外包圆的中心
        self.radius = radius
        self.left = left
        self.right = right
        self.points = points

class BallTree():
    def __init__(self,values,labels):
        self.values = values
        self.labels = labels
        if(len(self.values) == 0 ):
            raise Exception('Data For Ball-Tree Must Be Not empty.')
        self.root = self.build_BallTree()
        self.KNN_max_now_dist = np.inf
        self.KNN_result = [(None,self.KNN_max_now_dist)]

    def build_BallTree(self):
        data = np.column_stack((self.values,self.labels))
        return self.build_BallTree_core(data)

    def dist(self,point1,point2):
        return np.sqrt(np.sum((point1-point2)**2))

    #data:带标签的数据且已经排好序的
    def build_BallTree_core(self,data):
        if len(data) == 0:
            return None
        if len(data) == 1:
            return Ball(data[0,:-1],0.001,data,None,None)
        #当每个数据点完全一样时,全部归为一个球,及时退出递归,不然会导致递归层数太深出现程序崩溃
        data_disloc =  np.row_stack((data[1:],data[0]))
        if np.sum(data_disloc-data) == 0:
            return Ball(data[0, :-1], 1e-100, data, None, None)
        cur_center = np.mean(data[:,:-1],axis=0)     #当前球的中心
        dists_with_center = np.array([self.dist(cur_center,point) for point in data[:,:-1]])     #当前数据点到球中心的距离
        max_dist_index = np.argmax(dists_with_center)        #取距离中心最远的点,为生成下一级两个子球做准备,同时这也是当前球的半径
        max_dist = dists_with_center[max_dist_index]
        root = Ball(cur_center,max_dist,data,None,None)
        point1 = data[max_dist_index]

        dists_with_point1 = np.array([self.dist(point1[:-1],point) for point in data[:,:-1]])
        max_dist_index2 = np.argmax(dists_with_point1)
        point2 = data[max_dist_index2]            #取距离point1最远的点,至此,为寻找下一级的两个子球的准备工作搞定

        dists_with_point2 = np.array([self.dist(point2[:-1], point) for point in data[:, :-1]])
        assign_point1 = dists_with_point1 < dists_with_point2

        root.left = self.build_BallTree_core(data[assign_point1])
        root.right = self.build_BallTree_core(data[~assign_point1])
        return root    #是一个Ball

    def search_KNN(self,target,K):
        if self.root is None:
            raise Exception('KD-Tree Must Be Not empty.')
        if K > len(self.values):
            raise ValueError("K in KNN Must Be Greater Than Lenght of data")
        if len(target) !=len(self.root.center):
            raise ValueError("Target Must Has Same Dimension With Data")
        self.KNN_result = [(None,self.KNN_max_now_dist)]
        self.nums = 0
        self.search_KNN_core(self.root,target,K)
        return self.nums
        # print("calu_dist_nums:",self.nums)

    def insert(self,root_ball,target,K):
        for node in root_ball.points:
            self.nums += 1
            is_duplicate = [self.dist(node[:-1], item[0][:-1]) < 1e-4 and
                            abs(node[-1] - item[0][-1]) < 1e-4 for item in self.KNN_result if item[0] is not None]
            if np.array(is_duplicate, np.bool).any() and not allow_duplicate:
                continue
            distance = self.dist(target,node[:-1])
            if(len(self.KNN_result)<K):
                self.KNN_result.append((node,distance))
            elif distance < self.KNN_result[0][1]:
                self.KNN_result = self.KNN_result[1:] + [(node, distance)]
            self.KNN_result = sorted(self.KNN_result, key=lambda x: -x[1])


    #root是一个Ball
    def search_KNN_core(self,root_ball, target, K):
        if root_ball is None:
            return
        #在合格的超体空间(必须是最后一层的子空间)内查找更近的数据点
        if root_ball.left is None or root_ball.right is None:
            self.insert(root_ball, target, K)
        if abs(self.dist(root_ball.center,target)) <= root_ball.radius + self.KNN_result[0][1] : #or len(self.KNN_result) < K
            self.search_KNN_core(root_ball.left,target,K)
            self.search_KNN_core(root_ball.right,target,K)


if __name__ == '__main__':

    csv_path = "winequality-white.csv"
    data,lables,dim_label = load_data(csv_path)
    split_rate = 0.8 ; K=5
    train_num = int(len(data)*split_rate)
    print("train_num:",train_num)
    start1 = time.time()
    ball_tree = BallTree(data[:train_num], lables[:train_num])
    end1 = time.time()

    diff_all=0
    accuracy = 0
    search_all_time = 0
    calu_dist_nums = 0
    for index,target in enumerate(data[train_num:]):
        start2 = time.time()
        calu_dist_nums+=ball_tree.search_KNN(target, K)
        end2 = time.time()
        search_all_time += end2 - start2

        # for res in ball_tree.KNN_result:
        #     print("res:",res[0][:-1],res[0][-1],res[1])
        pred_label = Counter(node[0][-1] for node in ball_tree.KNN_result).most_common(1)[0][0]
        diff_all += abs(lables[index] - pred_label)
        if (lables[index] - pred_label) <= 0:
            accuracy += 1
        print("accuracy:", accuracy / (index + 1))
        print("Total:{},MSE:{:.3f}    {}--->{}".format(index + 1, (diff_all / (index + 1)), lables[index],
                                                   pred_label))


    print("BallTree构建时间：", end1 - start1)
    print("程序运行时间：", search_all_time/len(data[train_num:]))
    print("平均计算次数：", calu_dist_nums / len(data[train_num:]))
        #暴力KNN验证
        # KNN_res=[]
        # for index2,curData in enumerate(data[:train_num]):
        #     is_duplicate = [ball_tree.dist(curData,v[0])<1e-4 for v in KNN_res]
        #     if np.array(is_duplicate,np.bool).any() and not allow_duplicate:
        #         continue
        #     cur_dist = ball_tree.dist(curData,target)
        #     if len(KNN_res) < K:
        #        KNN_res.append((curData,lables[index2],cur_dist))
        #     elif cur_dist<KNN_res[0][2]:
        #         KNN_res = KNN_res[1:]+[(curData,lables[index2],cur_dist)]
        #     KNN_res=sorted(KNN_res, key=lambda x: -x[2])
        # pred_label2 = Counter(node[1] for node in KNN_res).most_common(1)[0][0]
        # for my_res in KNN_res:
        #     print("res:",my_res[0],my_res[1],my_res[2])
        # print("--------------{}--->{} vs {}------------------".format(lables[index],pred_label,pred_label2))