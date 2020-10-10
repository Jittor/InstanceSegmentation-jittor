
from jittor import nn,Module,init 
import jittor as jt

def sigmoid_focal_loss(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    class_range = jt.arange(1, num_classes+1, dtype=dtype).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = logits.sigmoid()
    term1 = (1 - p) ** gamma * jt.log(p)
    term2 = p ** gamma * jt.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def execute(self, logits, targets):
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
