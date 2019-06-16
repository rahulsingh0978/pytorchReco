import torch
from torch.autograd import Variable


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
        # create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
        # matrix multiplication
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)

class BaseModule(torch.nn.Module):
    """
    Base module for explicit matrix factorization.
    """

    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False):
        """
        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(BaseModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = torch.nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = torch.nn.Embedding(n_items, 1, sparse=sparse)
        self.user_embeddings = torch.nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = torch.nn.Embedding(n_items, n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(p=self.dropout_p)

        self.sparse = sparse

    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:
        user_bias + item_bias + user_embeddings.dot(item_embeddings)
        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices
        Returns
        -------
        preds : np.ndarray
            Predicted ratings.
        """

        # print ("inside forward")
        ues = self.user_embeddings(users)
        # print (ues.shape)
        uis = self.item_embeddings(items)
        # print (uis.shape)

        # print("self.user_biases(users) ",self.user_biases(users).shape)
        # print("self.item_biases(items) ",self.item_biases(items).shape)
        preds = self.user_biases(users)
        # print("preds 3" , preds.shape)
        preds += self.item_biases(items)
        # print("preds 4" , preds.shape)
        # test1 = self.dropout(ues)
        # test2 = self.dropout(uis)
        # print( "test 1" , test1.shape , "test2" ,test2.shape)
        # print(" mul ",(self.dropout(ues) * self.dropout(uis)).shape)
        # print("sum(1) ",sum(1))
        # print ( "preds " , (self.dropout(ues) * self.dropout(uis)).sum(1).view(1024,1).shape)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(1).view(preds.shape[0], preds.shape[1])
        # print ("final preds " ,preds)
        # print("preds " , preds)
        return preds

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)


class BPRModule(torch.nn.Module):

    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False,
                 model=BaseModule):
        super(BPRModule, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout_p = dropout_p
        self.sparse = sparse
        self.pred_model = model(
            self.n_users,
            self.n_items,
            n_factors=n_factors,
            dropout_p=dropout_p,
            sparse=sparse
        )

    def forward(self, users, items):
        assert isinstance(items, tuple), \
            'Must pass in items as (pos_items, neg_items)'
        # Unpack
        (pos_items, neg_items) = items
        pos_preds = self.pred_model(users, pos_items)
        neg_preds = self.pred_model(users, neg_items)
        return pos_preds - neg_preds

    def predict(self, users, items):
        return self.pred_model(users, items)

class DenseNet(torch.nn.Module):

    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False):
        # def __init__(self, n_users, n_items, n_factors=40,dropout_p=0,
        #            sparse=False, H1, D_out):
        """
        Simple Feedforward with Embeddings
        """
        super().__init__()
        # user and item embedding layers
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)
        # linear layers
        self.linear1 = torch.nn.Linear(n_factors * 2, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self, users, items):
        users_embedding = self.user_factors(users)
        items_embedding = self.item_factors(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([users_embedding, items_embedding], 1)
        h1_relu = F.relu(self.linear1(x))
        output_scores = self.linear2(h1_relu)
        return output_scores

    def predict(self, users, items):
        # return the score
        output_scores = self.forward(users, items)
        return output_scores