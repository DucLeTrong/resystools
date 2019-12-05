import argparse
import numpy as np
import torch

import model
import config
import evaluate
import data_utils
import torch.utils.data as data


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ItemPop")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Choose a dataset.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    path = args.path
    dataset = args.dataset
    train_data, test_data,test_data1, user_num ,item_num, train_mat = data_utils.load_all()
    test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
    test_loader = data.DataLoader(test_dataset,
		batch_size=100, shuffle=False, num_workers=0)
    item_score = np.array(train_mat.sum(axis=0, dtype=int)).flatten()
    HR, NDCG = [], []
    for user, item, label in test_loader:
        gt_item = item[0].item()
        # print(user)
        # print(int(user[1]))
        # print(test_data1[int(user[1])])
        # print(item_score[test_data1[int(user[1])]])
        # print(item_score[test_data1[int(user[1])]].argsort()[-10:][::-1])
        # print(type(item_score[test_data1[int(user[1])]].argsort()[-10:][::-1]))
        # print(type(test_data1))
        recommends = list(np.array(test_data1[int(user[1])])[item_score[test_data1[int(user[1])]].argsort()[-10:][::-1]])
        # print(recommends)
        HR.append(hit(gt_item,recommends))
        NDCG.append(ndcg(gt_item,recommends))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
        
if __name__ == "__main__":
    main()