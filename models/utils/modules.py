import torch.nn as nn
import torch
import torch.nn.functional as F

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):  # K, L, N
        """
        Bag-level classifier class

        Args:
            input_size (int): Input feature size.
            output_class (int): Number of output classes.
            dropout_v (float): Dropout probability for the V layer.
            nonlinear (bool): Whether to use non-linear activations.
        """
        super(BClassifier, self).__init__()
        if nonlinear:
            # Non-linear transformation
            self.lin = nn.Sequential(
                nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            # Identity transformation
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        # 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class,
                             kernel_size=input_size)

    def forward(self, feats, c):
        # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # Handle multiple classes without for loop
        # Sort class scores along the instance dimension, m_indices in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # Select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # Compute queries of critical instances, q_max in shape C x Q
        q_max = self.q(m_feats)
        # Compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))
        # Normalize attention scores, A in shape N x C,
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        # Compute bag representation, B in shape C x V
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B

class MLP(nn.Module):
    def __init__(self,in_s,out_s):
        super(MLP,self).__init__()
        self.mlp=nn.Sequential( nn.Linear(int(in_s),int(in_s/2)),
                                nn.ELU(),
                                nn.Linear(int(in_s/2),int(out_s))
                                )
    def forward(self,input):
        return self.mlp(input)

class GatedLinearUnit(nn.Module):
    def __init__(self, in_s,out_s,norm):
        super(GatedLinearUnit, self).__init__()
        self.mlp =MLP(in_s,out_s)
        self.sigmoid = nn.Sigmoid()
        self.norm=norm
        if self.norm:
            self.norm=nn.LayerNorm(normalized_shape=out_s)

    def forward(self, inputs):
        x=inputs * self.sigmoid(self.mlp(inputs))
        if self.norm:
            x=self.norm(x)
        return x

def init(model, state_dict_weights):
    if state_dict_weights is not None:
        try:
            model.load_state_dict(state_dict_weights, strict=False)
        except:
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            model.load_state_dict(state_dict_weights, strict=False)
    return model

