import kornia
import torch
import torch.nn


class DiceLoss(nn.Module):

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

    math:
        B(y, y') = (1/n) * |y' - y| if |y'-y| <= c
        B(y, y') = (1/n) * ( (y'-y)^2 + c^2 ) / 2*c othwerwise

        c = 1/5*max(|y'-y|)

    [1] http://cs231n.stanford.edu/reports/2017/pdfs/203.pdf
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
    '''
    the loss functions inside take care of sigmoiding the input and taking mean
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
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, input, target):
        input_sig = torch.sigmoid(input)
        loss = torch.sqrt(self.mse(input_sig, target) + self.eps)
        return loss
