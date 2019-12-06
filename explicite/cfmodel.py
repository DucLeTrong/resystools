import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
class CF(object):
    def __init__(self):
        pass
    def normalize_Y(self):
        pass
    def fit(self):
        pass
    def pred(self):
        pass
    def evaluate(self):
        pass
    

class MF(CF):
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
                learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 0):
        self.Y_data = Y_data
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = user_based
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(Y_data[:, 0])) + 1 
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        
        if Xinit is None: 
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = Xinit 
        
        if Winit is None: 
            self.W = np.random.randn(K, self.n_users)
        else: 
            self.W = Winit
        
        # item biases
        self.b = np.random.randn(self.n_items)
        self.d = np.random.randn(self.n_users)
        #self.all_users = self.Y_data[:,0] # all users (may be duplicated)
        self.n_ratings = Y_data.shape[0]
#         self.mu = np.mean(Y_data[:, 2])
        self.mu = 0


    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        else:
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = self.Y_data[:, user_col] 
        self.muu = np.zeros((n_objects,))
        for n in xrange(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, item_col] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.muu[n] = m
            # normalize
            self.Y_data[ids, 2] = ratings - m
            
            
    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data[i, 0]), int(self.Y_data[i, 1]), self.Y_data[i, 2]
            L += 0.5*(self.X[m, :].dot(self.W[:, n]) + self.b[m] + self.d[n] + self.mu - rate)**2
            
        # regularization, don't ever forget this 
        L /= self.n_ratings
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro') + \
                        np.linalg.norm(self.b) + np.linalg.norm(self.d))
        return L 

    
    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user n, and the corresponding ratings
        """
        # y = self.Y_data_n[:,0] # all users (may be duplicated)
        # item indices rated by user_id
        # we need to +1 to user_id since in the rate_matrix, id starts from 1 
        # while index in python starts from 0
        ids = np.where(self.Y_data[:,0] == user_id)[0] 
        item_ids = self.Y_data[ids, 1].astype(np.int32) # index starts from 0 
        ratings = self.Y_data[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item m and get the corresponding ratings
        """
        ids = np.where(self.Y_data[:,1] == item_id)[0] 
        user_ids = self.Y_data[ids, 0].astype(np.int32)
        ratings = self.Y_data[ids, 2]
        return (user_ids, ratings)
        
    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            
            Wm = self.W[:, user_ids]
            dm = self.d[user_ids]
            xm = self.X[m, :]
            
            error = xm.dot(Wm) + self.b[m] + dm + self.mu - ratings 
            
            grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam*xm
            grad_bm = np.sum(error)/self.n_ratings + self.lam*self.b[m]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
            self.b[m] -= self.learning_rate*grad_bm
    
    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            bn = self.b[item_ids]
            wn = self.W[:, n]
            
            error = Xn.dot(wn) + bn + self.mu + self.d[n] - ratings
            grad_wn = Xn.T.dot(error)/self.n_ratings + self.lam*wn
            grad_dn = np.sum(error)/self.n_ratings + self.lam*self.d[n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
            self.d[n] -= self.learning_rate*grad_dn
    
    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_data)
                print 'iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train
    
    
    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        if you need the un
        """
        u = int(u)
        i = int(i)
        if self.user_based == 1:
            bias = self.muu[u]
        else:
            bias = self.muu[i]
        
        pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u] + bias

        return max(0, min(5, pred))
        
    
    def pred_for_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()              
        
        y_pred = self.X.dot(self.W[:, user_id])
        predicted_ratings= []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings
    
    def evaluate_rmse(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in xrange(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE

    def evaluate_mae(self, rate_test):
        n_tests = rate_test.shape[0]
        SAE = 0 # squared error
        for n in xrange(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SAE += abs(pred - rate_test[n, 2])

        MAE = SE/n_tests
        return MAE

class UserBase(CF):
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        eps = 1e-6
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
        
    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
        
    
    def __pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2: 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others 
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:] 
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
    
    def evaluate_rmse(self, rate_test)
        n_tests = rate_test.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
            SE += (pred - rate_test[n, 2])**2 
        return np.sqrt(SE/n_tests)
    
    def evaluate_mae(self, rate_test)
        n_tests = rate_test.shape[0]
        SAE = 0
        for n in range(n_tests):
            pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
            SAE += abs(pred - rate_test[n, 2])
        return SAE/n_tests
    
    def recommend(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been rated by u yet. 
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    recommended_items.append(i)
        
        return recommended_items 
    
    def recommend2(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been rated by u yet. 
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
    
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    recommended_items.append(i)
        
        return recommended_items 

    def print_recommendation(self):
        """
        print all items which should be recommended for each user 
        """
        # print 'Recommendation: '
        for u in xrange(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print('    Recommend item(s):', recommended_items, 'for user', u)
            else: 
                print('    Recommend item', u, 'for user(s) : ', recommended_items)

class ItemBase(CF):
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 0):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        eps = 1e-6
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
        
    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
        
    
    def __pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2: 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others 
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:] 
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
            
    def evaluate_rmse(self, rate_test)
        n_tests = rate_test.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
            SE += (pred - rate_test[n, 2])**2 
        return np.sqrt(SE/n_tests)
    
    def evaluate_mae(self, rate_test)
        n_tests = rate_test.shape[0]
        SAE = 0
        for n in range(n_tests):
            pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
            SAE += abs(pred - rate_test[n, 2])
        return SAE/n_tests
    # def recommend(self, u):
    #     """
    #     Determine all items should be recommended for user u.
    #     The decision is made based on all i such that:
    #     self.pred(u, i) > 0. Suppose we are considering items which 
    #     have not been rated by u yet. 
    #     """
    #     ids = np.where(self.Y_data[:, 0] == u)[0]
    #     items_rated_by_u = self.Y_data[ids, 1].tolist()              
    #     recommended_items = []
    #     for i in xrange(self.n_items):
    #         if i not in items_rated_by_u:
    #             rating = self.__pred(u, i)
    #             if rating > 0: 
    #                 recommended_items.append(i)
        
    #     return recommended_items 
    
    # def recommend2(self, u):
    #     """
    #     Determine all items should be  recommended for user u.
    #     The decision is made based on all i such that:
    #     self.pred(u, i) > 0. Suppose we are considering items which 
    #     have not been rated by u yet. 
    #     """
    #     ids = np.where(self.Y_data[:, 0] == u)[0]
    #     items_rated_by_u = self.Y_data[ids, 1].tolist()              
    #     recommended_items = []
    
    #     for i in xrange(self.n_items):
    #         if i not in items_rated_by_u:
    #             rating = self.__pred(u, i)
    #             if rating > 0: 
    #                 recommended_items.append(i)
        
    #     return recommended_items 

        