from torch import nn
import torch
from torch.optim import SGD
from objectives import calculate_dp_obj, calculate_eo_obj
from evaluation import calculate_acc, calculate_dp, calculate_eo, calculate_eodds
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from pgd import solve_min_norm


class Base_clf(nn.Module):

	def __init__(self, in_features, list_hidden_units):
		super().__init__()
		
		list_hidden_layers = []
		for num_hidden_units in list_hidden_units:
			list_hidden_layers.append(nn.Linear(in_features, num_hidden_units))
			list_hidden_layers.append(nn.ReLU())
			in_features = num_hidden_units
		list_hidden_layers.append(nn.Linear(in_features, 1))

		self.model = nn.Sequential(*list_hidden_layers)

	def forward(self, X):
		Y_hat = self.model(X)

		return Y_hat.squeeze()

class Fair_learner(nn.Module):
	def __init__(self, in_features, list_hidden_units, 
			  feature_column, target_column, group_column, 
			  lr, alpha_lr, fix_alpha=None):
		super().__init__()
		self.base_clf = Base_clf(in_features, list_hidden_units)
		self.optimizer = SGD(self.base_clf.parameters(), lr)

		self.alpha_lr = alpha_lr

		self.feature_column = feature_column
		self.target_column = target_column
		self.group_column = group_column

		self.fix_alpha = fix_alpha
	
	def pareto_update(self, batch_df, reg=True):

		self.base_clf.train()

		X = torch.tensor(batch_df[self.feature_column].to_numpy(), dtype=torch.float)
		Y = torch.tensor(batch_df[self.target_column].to_numpy(), dtype=torch.float)
		G = torch.tensor(batch_df[self.group_column].to_numpy(), dtype=torch.float)


		Y_hat = self.base_clf(X).squeeze()

		self.optimizer.zero_grad()
		acc_obj = F.binary_cross_entropy_with_logits(Y_hat, Y)

		race_dp_obj = calculate_dp_obj(Y_hat, Y, G[:, 0])
		sex_dp_obj = calculate_dp_obj(Y_hat, Y, G[:, 1])

		race_eo_obj = calculate_eo_obj(Y_hat, Y, G[:, 0])
		sex_eo_obj = calculate_eo_obj(Y_hat, Y, G[:, 1])
		obj_list = [acc_obj, race_dp_obj, sex_dp_obj, race_eo_obj, sex_eo_obj]

		obj_grad_list = []
		for i, obj in enumerate(obj_list):
			obj.backward(retain_graph=True)
			obj_grads = [param.grad.view(-1) for param in self.base_clf.parameters()]
			# obj_grad_list.append(torch.cat(obj_grads))
			# obj_grad_list.append(torch.cat(obj_grads) / (torch.norm(torch.cat(obj_grads)) * obj.detach()) )
			# if i == 0:
				# obj_grad_list.append(torch.cat(obj_grads) / (torch.norm(torch.cat(obj_grads))))
			# else:
			obj_grad_list.append(torch.cat(obj_grads) / torch.norm(torch.cat(obj_grads)))
		alpha, min_norm = solve_min_norm(torch.stack(obj_grad_list), self.alpha_lr)
		
		self.optimizer.zero_grad()
		acc_obj = F.binary_cross_entropy_with_logits(Y_hat, Y)

		race_dp_obj = calculate_dp_obj(Y_hat, Y, G[:, 0])
		sex_dp_obj = calculate_dp_obj(Y_hat, Y, G[:, 1])

		race_eo_obj = calculate_eo_obj(Y_hat, Y, G[:, 0])
		sex_eo_obj = calculate_eo_obj(Y_hat, Y, G[:, 1])


		obj_tensor = torch.stack([2.8 * acc_obj, race_dp_obj, sex_dp_obj, race_eo_obj, sex_eo_obj])

		total_obj = (alpha * obj_tensor).sum()
		total_obj.backward()
		self.optimizer.step()

	def evaluate_model(self, test_df):

		self.base_clf.eval()
		
		X = torch.tensor(test_df[self.feature_column].to_numpy(), dtype=torch.float)
		Y = torch.tensor(test_df[self.target_column].to_numpy(), dtype=torch.float)
		G = torch.tensor(test_df[self.group_column].to_numpy(), dtype=torch.float)

		with torch.no_grad():
			Y_hat = self.base_clf(X).squeeze()

			acc_obj = F.binary_cross_entropy_with_logits(Y_hat, Y).item()
			# f_obj = calculate_dp_obj(Y_hat, Y, G).item()
			f_obj = 0

			acc = calculate_acc(Y_hat, Y)
			auc = roc_auc_score(Y.numpy(), torch.sigmoid(Y_hat).numpy())
			g1_dp = calculate_dp(Y_hat, Y, G[:, 0])
			g1_eo = calculate_eo(Y_hat, Y, G[:, 0])
			g1_eodds = calculate_eodds(Y_hat, Y, G[:, 0])

			g2_dp = calculate_dp(Y_hat, Y, G[:, 1])
			g2_eo = calculate_eo(Y_hat, Y, G[:, 1])
			g2_eodds = calculate_eodds(Y_hat, Y, G[:, 1])

		return acc_obj, f_obj, acc, auc, g1_dp, g1_eo, g1_eodds, g2_dp, g2_eo, g2_eodds
