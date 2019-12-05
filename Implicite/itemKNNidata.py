import numpy as np
import time
import math
import data_utils

def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    matrix = np.zeros((num_users,num_items))
    for line in open(filename):
        user,item,_,_ = line.split()
        user = int(user)
        item = int(item)
        count = 1.0
        matrix[user-1,item-1] = count
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1-t0)
    return  matrix

class ItemCF:

    def __init__(self,traindata,testdata):
        self.traindata = traindata
        self.testdata = testdata
        self.num_users = traindata.shape[0]
        self.num_items = traindata.shape[1]

    def ItemSimilarity(self):
        t0 = time.time()
        train = self.traindata
        num_items = self.num_items
        self.item_similarity = np.zeros((num_items,num_items))

        for i in range(num_items):
            r_i = train[:,i]
            self.item_similarity[i,i] = 0
            for j in range(i+1,num_items):
                r_j = train[:,j]
                num = np.dot(r_i.T , r_j)
                denom = np.linalg.norm(r_i) * np.linalg.norm(r_j)
                if denom == 0:
                    cos = 0
                else:
                    cos = num / denom
                self.item_similarity[i,j] = cos
                self.item_similarity[j,i] = cos
        self.item_neighbor = np.argsort(-self.item_similarity)
        t1 = time.time()
        print 'Finished calculating similarity matrix in %f seconds' % (t1-t0)

    def ItemCFPrediction(self,user_id,item_id,kNN):
        # predict the preference of a user for an item using kNN item-based collaborative filtering
        # r_ui = \sum_{j \in N^k(i)} r_uj \times w_ij
        pred_score = 0

        train = self.traindata
        similarity = self.item_similarity

        # find the user's rating history
        neigh_of_item = self.item_neighbor[item_id]
        neigh = neigh_of_item[0:kNN]
        for neigh_item in neigh:
            pred_score = pred_score + train[user_id,neigh_item] * similarity[item_id,neigh_item]
        return pred_score

    def ItemCFPrediction(self, user_id, item_id, KNN,list_test_item):
        train = self.traindata
        similarity = self.item_similarity
        #fghjkl
        neigh_of_item = self.item_neighbor[item_id][list_test_item]
        neigh = neigh_of_item[0:kNN]
        for neigh_item in neigh:
            pred_score = pred_score + train[user_id,neigh_item] * similarity[item_id,neigh_item]
        return pred_score



    def Evaluate(self,kNN,top_N):
        precision = 0
        recall = 0
        user_count = 0

        train = self.traindata
        test = self.testdata
        num_users = self.num_users
        num_items = self.num_items

        idcg = np.zeros(num_users)
        dcg  = np.zeros(num_users)
        ndcg = np.zeros(num_users)
        map = np.zeros(num_users)

        for u in range(num_users):
            r_u_test = test[u]
            test_items = np.nonzero(r_u_test)
            test_items_idx = test_items[0]
            if len(test_items_idx) == 0:    # if this user does not possess any rating in the test set, skip the evaluate procedure
                continue
            else:
                r_u_train = train[u]
                train_items = np.nonzero(r_u_train)
                train_items_idx = train_items[0]    # items user u rated in the train data set, which we do not need to predict
                pred_item_idx = np.setdiff1d(range(num_items),train_items_idx)
                pred_sore = np.zeros(num_items)
                for item in pred_item_idx:
                    pred_sore[item] = self.ItemCFPrediction(u,item,kNN)
                rec_cand = np.argsort(-pred_sore)
                rec_list = rec_cand[0:top_N]
                hit_set = np.intersect1d(rec_list,test_items_idx)
                precision = precision + len(hit_set) / (top_N * 1.0)
                recall = recall + len(hit_set) / (len(test_items_idx) * 1.0)
                user_count = user_count + 1

                # calculate the ndcg and map measure
                if len(hit_set) != 0:
                    rel_list = np.zeros((len(rec_list)))
                    rank = 0.0 # to calculate the idcg measure
                    for item in hit_set:
                        rec_of_i = list(rec_list)
                        item_rank = rec_of_i.index(item)    # relevant items in the rec_of_i, to calculate dcg measure
                        rel_list[item_rank] = 1
                        dcg[u] = dcg[u] + 1.0 / math.log(item_rank+2,2)
                        idcg[u] = idcg[u] + 1.0 / math.log(rank+2,2)
                        map[u] = map[u] + (rank+1) / (item_rank + 1.0)
                        rank = rank + 1
                    ndcg[u] = dcg[u] / (idcg[u] * 1.0)
                    map[u] = map[u] / len(hit_set)

        ndcg = sum(ndcg) / user_count
        map = sum(map) / user_count
        precision = precision / (user_count * 1.0)
        recall = recall / (user_count * 1.0)

        return precision,recall,ndcg,map

def test():
    kNN = [80]
    top_N = [5,10,15,20]
    # train_data = load_matrix('ua.base',943,1682)
    # test_data = load_matrix('ua.test',943,1682)
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
    train_matrix = np.zeros((user_num,item_num))
    for i in train_data:
        train_matrix[i[0],i[1]] = 1
    kNNItemCF = ItemCF(train_data,test_data)
    kNNItemCF.ItemSimilarity()
    print "%7s %7s %20s%20s%20s%20s" % ('kNN','top_N',"precision",'recall','ndcg','map')
    for k in kNN:
        for N in top_N:
            precision,recall,ndcg,map = kNNItemCF.Evaluate(k,N)
            print "%5d%5d%19.3f%%%19.3f%%19.3f%%19.3f%%" % (k,N,precision*100,recall*100,ndcg*100,map*100)

if __name__=='__main__':
    test()