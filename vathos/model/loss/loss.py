import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

        According to [1], we compute the Sørensen-Dice Coefficient as follows:

        .. math::

            \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

        where:
        - :math:`X` expects to be the scores of each class.
        - :math:`Y` expects to be the one-hot tensor with the class labels.

        the loss, is finally computed as:

        .. math::

            \text{loss}(x, class) = 1 - \text{Dice}(x, class)

        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        input_sig = torch.sigmoid(input)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_sig * target, dims)
        cardinality = torch.sum(input_sig + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(torch.tensor(1.) - dice_score)


class TverskyLoss(nn.Module):
    r"""Performs Tversky Loss on Logits

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Reference:
        [1] https://kornia.readthedocs.io/en/latest/losses.html
    """

    def __init__(self, alpha: float, beta: float) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        input_sig = torch.sigmoid(input)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_sig * target, dims)
        fps = torch.sum(input_sig * (torch.tensor(1.) - target), dims)
        fns = torch.sum((torch.tensor(1.) - input_sig) * target, dims)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        return torch.mean(torch.tensor(1.) - tversky_loss)


class BCEDiceLoss(nn.Module):
    r"""Performs BCE and Dice Loss and adds them both

    loss = bce_loss + 2 * dice_loss
    """

    def __init__(self) -> None:
        super(BCEDiceLoss, self).__init__()
        self.eps = 1e-6
        self.dice_loss = DiceLoss()

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        bce_loss = F.binary_cross_entropy_with_logits(input, target)
        dice_loss = self.dice_loss(input, target)

        loss = bce_loss + 2*dice_loss

        return loss


class BCETverskyLoss(nn.Module):
    r"""Performs BCE and Tversky Loss and adds them both

    loss = bce_loss + 2 * tversky_loss
    """

    def __init__(self) -> None:
        super(BCETverskyLoss, self).__init__()
        self.eps = 1e-6
        self.tversky_loss = TverskyLoss(0.6, 0.5)

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        bce_loss = F.binary_cross_entropy_with_logits(input, target)
        tversky_loss = self.tversky_loss(input, target)

        loss = bce_loss + 2*tversky_loss

        return loss


class BerHuLoss(nn.Module):
    r'''
    Implementation of the BerHu Loss from [1]

    .. math::
            B(y, y') = (1/n) * |y' - y| if |y'-y| <= c

            B(y, y') = (1/n) * ( (y'-y)^2 + c^2 ) / 2*c othwerwise

            c = 1/5*max(|y'-y|)

    [1] http://cs231n.stanford.edu/reports/2017/pdfs/203.pdf

    [2] https://arxiv.org/abs/1207.6868
    '''

    def __init__(self, threshold: float = 1./5):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input_sig = torch.sigmoid(input)

        diff = torch.abs(target - input)
        C = self.threshold * torch.max(diff).item()

        # if -|y'-y| >= -c then |y'-y| else 0
        l_eq = -F.threshold(-diff, -C, 0.)
        # if diff^2 - c^2 > 0 then diff^2-c^2 + 2c^2 / 2c else -2c^2 + 2c^2 / 2c
        l_other = (F.threshold(diff**2 - C**2, 0., -2*C**2) + 2*C**2) / 2*C

        loss = l_eq + l_other

        loss = torch.mean(loss)

        return loss


class GradLoss(nn.Module):
    r"""Performs Gradient Loss

    The Image XY Gradients are computed for input and target and the mean L1Loss between these
    gradients is returned
    """

    def __init__(self):
        super(GradLoss, self).__init__()

        # B, C, H, W  -> B, C, 2, H, W
        self.input_grad = kornia.filters.SpatialGradient()
        self.target_grad = kornia.filters.SpatialGradient()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_sig = torch.sigmoid(input)
        input_grads = self.input_grad(input_sig)
        target_grads = self.target_grad(target)

        loss = torch.mean(torch.abs(target_grads - input_grads))

        return loss


class SSIMLoss(nn.Module):
    r"""Performs SSIM Loss

    window sizes uses are 5x5 and 11x11

    we tried adding other window sizes too, but there wasn't a significant benefit

    .. note::
        we do ssim loss for various window sizes, add them and return the mean
    """

    def __init__(self):
        super(SSIMLoss, self).__init__()

        self.ssim_loss_5x5 = kornia.losses.SSIM(5, reduction='none')
        self.ssim_loss_11x11 = kornia.losses.SSIM(11, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_sig = torch.sigmoid(input)

        loss_5 = self.ssim_loss_5x5(input_sig, target)
        loss_11 = self.ssim_loss_11x11(input_sig, target)

        return torch.mean(loss_5) + torch.mean(loss_11)


class RMSEwSSIMLoss(nn.Module):
    r'''Performs RMSE and SSIM Loss

    loss = :math:`\sqrt{\text{ssim_loss} + 4\times \text{rmse_loss}}`
    '''

    def __init__(self):
        super(RMSEwSSIMLoss, self).__init__()

        self.ssim_loss = SSIMLoss()
        self.rmse_loss = RMSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = torch.sqrt(self.ssim_loss(input, target)) + \
            4*self.rmse_loss(input, target)

        return loss


def iou(outputs: torch.Tensor, labels: torch.Tensor):
    eps: float = 1e-6

    dims = (1, 2, 3)

    outputs = torch.sigmoid(outputs)
    intersection = torch.sum((outputs * labels), dims)
    union = torch.sum((outputs + labels), dims)

    iou = intersection / (union + eps)

    miou = torch.mean(iou)

    return miou


def rmse(outputs: torch.Tensor, labels: torch.Tensor):
    rmse_loss = RMSELoss()

    loss = rmse_loss(outputs, labels)
    return loss


class RMSELoss(nn.Module):
    r"""Performs RMSE Loss

    we simply sigmoid the input, pass it through `nn.MSELoss` and then do a `torch.sqrt` on it
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, input, target):
        input_sig = torch.sigmoid(input)
        loss = torch.sqrt(self.mse(input_sig, target) + self.eps)
        return loss
