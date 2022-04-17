import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from common import class_spec_nms, get_fpn_location_coords, nms
from torch import nn
from torch.nn import functional as F

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")


class RPNPredictionNetwork(nn.Module):
    """
    RPN prediction network that accepts FPN feature maps from different levels
    and makes two predictions for every anchor: objectness and box deltas.

    Faster R-CNN typically uses (p2, p3, p4, p5) feature maps. We will exclude
    p2 for have a small enough model for Colab.

    Conceptually this module is quite similar to `FCOSPredictionNetwork`.
    """

    def __init__(
        self, in_channels: int, stem_channels: List[int], num_anchors: int = 3
    ):
        """
        Args:
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
            num_anchors: Number of anchor boxes assumed per location (say, `A`).
                Faster R-CNN without an FPN uses `A = 9`, anchors with three
                different sizes and aspect ratios. With FPN, it is more common
                to have a fixed size dependent on the stride of FPN level, hence
                `A = 3` is default - with three aspect ratios.
        """
        super().__init__()

        self.num_anchors = num_anchors
        # Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. 
        stem_rpn = []
        conv = nn.Conv2d(in_channels, stem_channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.normal_(conv.weight, mean=0, std=0.01)
        nn.init.zeros_(conv.bias)
        
        stem_rpn.append(conv)
        stem_rpn.append(nn.ReLU())
        
        for i in range(len(stem_channels)-1):
            conv = nn.Conv2d(stem_channels[i], stem_channels[i+1], kernel_size=3, stride=1, padding=1, bias=True)
            nn.init.normal_(conv.weight, mean=0, std=0.01)
            nn.init.zeros_(conv.bias)
            stem_rpn.append(conv)
            stem_rpn.append(nn.ReLU())

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_rpn = nn.Sequential(*stem_rpn)
        # Create TWO 1x1 conv layers for individually to predict objectness and box deltas for every anchor, at every location.
        self.pred_obj = None  # Objectness conv
        self.pred_box = None  # Box regression conv

        obj_conv = nn.Conv2d(stem_channels[-1], self.num_anchors, 1)
        nn.init.normal_(obj_conv.weight, mean=0, std=0.01)
        nn.init.zeros_(obj_conv.bias)
        self.pred_obj = obj_conv
        
        box_conv = nn.Conv2d(stem_channels[-1], self.num_anchors * 4, 1)
        nn.init.normal_(box_conv.weight, mean=0, std=0.01)
        nn.init.zeros_(box_conv.bias)
        self.pred_box = box_conv

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict desired quantities for every anchor
        at every location. Format the output tensors such that feature height,
        width, and number of anchors are collapsed into a single dimension (see
        description below in "Returns" section) this is convenient for computing
        loss and perforning inference.

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}.
                Each tensor will have shape `(batch_size, fpn_channels, H, W)`.

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Objectness logits:     `(batch_size, H * W * num_anchors)`
            2. Box regression deltas: `(batch_size, H * W * num_anchors, 4)`
        """

        object_logits = {}
        boxreg_deltas = {}
        
        pred_obj = self.pred_obj
        pred_box = self.pred_box
       
        boxreg_deltas["p3"] = pred_box(self.stem_rpn(feats_per_fpn_level["p3"])).permute([0,2,3,1])
        boxreg_deltas["p4"] = pred_box(self.stem_rpn(feats_per_fpn_level["p4"])).permute([0,2,3,1])
        boxreg_deltas["p5"] = pred_box(self.stem_rpn(feats_per_fpn_level["p5"])).permute([0,2,3,1])
        
        
        object_logits["p3"] = pred_obj(self.stem_rpn(feats_per_fpn_level["p3"])).permute([0,2,3,1])
        object_logits["p4"] = pred_obj(self.stem_rpn(feats_per_fpn_level["p4"])).permute([0,2,3,1])
        object_logits["p5"] = pred_obj(self.stem_rpn(feats_per_fpn_level["p5"])).permute([0,2,3,1])
        
        object_logits["p3"] = object_logits["p3"].reshape(boxreg_deltas["p3"].shape[0], -1)
        object_logits["p4"] = object_logits["p4"].reshape(boxreg_deltas["p3"].shape[0], -1)
        object_logits["p5"] = object_logits["p5"].reshape(boxreg_deltas["p3"].shape[0], -1)
        
        boxreg_deltas["p3"] = boxreg_deltas["p3"].reshape(boxreg_deltas["p3"].shape[0], -1, 4)
        boxreg_deltas["p4"] = boxreg_deltas["p4"].reshape(boxreg_deltas["p3"].shape[0], -1, 4)
        boxreg_deltas["p5"] = boxreg_deltas["p5"].reshape(boxreg_deltas["p3"].shape[0], -1, 4)

        return [object_logits, boxreg_deltas]


@torch.no_grad()
def generate_fpn_anchors(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    stride_scale: int,
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
):
    """
    Generate multiple anchor boxes at every location of FPN level. Anchor boxes
    should be in XYXY format and they should be centered at the given locations.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H, W is the size of FPN feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        stride_scale: Size of square anchor at every FPN levels will be
            `(this value) * (FPN level stride)`. Default is 4, which will make
            anchor boxes of size (32x32), (64x64), (128x128) for FPN levels
            p3, p4, and p5 respectively.
        aspect_ratios: Anchor aspect ratios to consider at every location. We
            consider anchor area to be `(stride_scale * FPN level stride) ** 2`
            and set new width and height of anchors at every location:
                new_width = sqrt(area / aspect ratio)
                new_height = area / new_width

    Returns:
        TensorDict
            Dictionary with same keys as `locations_per_fpn_level` and values as
            tensors of shape `(HWA, 4)` giving anchors for all locations
            per FPN level, each location having `A = len(aspect_ratios)` anchors.
            All anchors are in XYXY format and their centers align with locations.
    """

    # Set these to `(N, A, 4)` Tensors giving anchor boxes in XYXY format.
    anchors_per_fpn_level = {
        level_name: None for level_name, _ in locations_per_fpn_level.items()
    }

    for level_name, locations in locations_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        # List of `A = len(aspect_ratios)` anchor boxes.
        anchor_boxes = []
        for aspect_ratio in aspect_ratios:
            w = level_stride * stride_scale
            area = w ** 2
            new_w = (area / aspect_ratio) ** 0.5
            new_h = area / new_w
            
            l = locations[:, 0].shape[0]
            x1 = locations[:, 0].reshape(l, 1) - 0.5 * new_w
            y1 = locations[:, 1].reshape(l, 1) - 0.5 * new_h
            x2 = locations[:, 0].reshape(l, 1) + 0.5 * new_w
            y2 = locations[:, 1].reshape(l, 1) + 0.5 * new_h
            cord = torch.cat((x1, y1, x2, y2), dim=1)

        # shape: (A, H * W, 4)
        anchor_boxes = torch.stack(anchor_boxes)
        # Bring `H * W` first and collapse those dimensions.
        anchor_boxes = anchor_boxes.permute(1, 0, 2).contiguous().view(-1, 4)
        anchors_per_fpn_level[level_name] = anchor_boxes

    return anchors_per_fpn_level


@torch.no_grad()
def iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection-over-union (IoU) between pairs of box tensors. Input
    box tensors must in XYXY format.

    Args:
        boxes1: Tensor of shape `(M, 4)` giving a set of box co-ordinates.
        boxes2: Tensor of shape `(N, 4)` giving another set of box co-ordinates.

    Returns:
        torch.Tensor
            Tensor of shape (M, N) with `iou[i, j]` giving IoU between i-th box
            in `boxes1` and j-th box in `boxes2`.
    """
    b1x1, b1y1, b1x2, b1y2 = torch.split(boxes1, 1, dim=1)
    b2x1, b2y1, b2x2, b2y2 = torch.split(boxes2, 1, dim=1)
    
    xA = torch.max(b1x1, b2x1.T)
    yA = torch.max(b1y1, b2y1.T)
    xB = torch.min(b1x2, b2x2.T)
    yB = torch.min(b1y2, b2y2.T)
    
    b1Area = (b1x2 - b1x1) * (b1y2 - b1y1)
    b2Area = (b2x2 - b2x1) * (b2y2 - b2y1)
    
    interW = torch.clamp((xB - xA), min=0)
    interH = torch.clamp((yB - yA), min=0)
    interArea = interW * interH

    iou = interArea/(b1Area + b2Area.T - interArea)
    return iou


@torch.no_grad()
def rcnn_match_anchors_to_gt(
    anchor_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: Tuple[float, float],
) -> TensorDict:
    """
    Match anchor boxes (or RPN proposals) with a set of GT boxes. Anchors having
    high IoU with any GT box are assigned "foreground" and matched with that box
    or vice-versa.

    NOTE: This function is NOT BATCHED. Call separately for GT boxes per image.

    Args:
        anchor_boxes: Anchor boxes (or RPN proposals). A combined tensor of some
            shape `(N, 4)` where `N` represents:
                - in first stage: total anchors from all FPN levels.
                - in second stage: set of RPN proposals from first stage.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.
        iou_thresholds: Tuple of (low, high) IoU thresholds, both in [0, 1]
            giving thresholds to assign foreground/background anchors.
    """

    # Filter empty GT boxes:
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]

    # If no GT boxes are available, match all anchors to background and return.
    if len(gt_boxes) == 0:
        fake_boxes = torch.zeros_like(anchor_boxes) - 1
        fake_class = torch.zeros_like(anchor_boxes[:, [0]]) - 1
        return torch.cat([fake_boxes, fake_class], dim=1)

    # Match matrix => pairwise IoU of anchors (rows) and GT boxes (columns).
    match_matrix = iou(anchor_boxes, gt_boxes[:, :4])

    # Find matched ground-truth instance per anchor:
    match_quality, matched_idxs = match_matrix.max(dim=1)
    matched_gt_boxes = gt_boxes[matched_idxs]

    # Set boxes with low IoU threshold to background (-1).
    matched_gt_boxes[match_quality <= iou_thresholds[0]] = -1

    # Set remaining boxes to neutral (-1e8).
    neutral_idxs = (match_quality > iou_thresholds[0]) & (
        match_quality < iou_thresholds[1]
    )
    matched_gt_boxes[neutral_idxs, :] = -1e8
    return matched_gt_boxes


def rcnn_get_deltas_from_anchors(
    anchors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Get box regression deltas that transform `anchors` to `gt_boxes`. These
    deltas will become GT targets for box regression. Unlike FCOS, the deltas
    are in `(dx, dy, dw, dh)` format that represent offsets to anchor centers
    and scaling factors for anchor size. Box regression is only supervised by
    foreground anchors. If GT boxes are "background/neutral", then deltas
    must be `(-1e8, -1e8, -1e8, -1e8)` (just some LARGE negative number).

    Follow Slide 68:
        https://web.eecs.umich.edu/~justincj/slides/eecs498/WI2022/598_WI2022_lecture13.pdf

    Args:
        anchors: Tensor of shape `(N, 4)` giving anchors boxes in XYXY format.
        gt_boxes: Tensor of shape `(N, 4)` giving matching GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving anchor deltas.
    """
    deltas = None
    # XYXY for gt box
    gt_x1 = gt_boxes[:,0]
    gt_y1 = gt_boxes[:,1]
    gt_x2 = gt_boxes[:,2]
    gt_y2 = gt_boxes[:,3]
    
    anchors_x1 = anchors[:,0]
    anchors_y1 = anchors[:,1]
    anchors_x2 = anchors[:,2]
    anchors_y2 = anchors[:,3]
    
    # width and height for gt box and anchors
    bw = gt_x2 - gt_x1
    bh = gt_y2 - gt_y1
    pw = anchors_x2 - anchors_x1
    ph = anchors_y2 - anchors_y1
    
    # center of gt box and anchors
    bx = (bw) / 2 + gt_x1
    by = (bh) / 2 + gt_y1
    px = (pw) / 2 + anchors_x1
    py = (ph) / 2 + anchors_y1
    
    # delta
    tx = (bx - px) / pw
    ty = (by - py) / ph
    tw = torch.log(bw / pw)
    th = torch.log(bh / ph)
    deltas = torch.zeros_like(anchors)
    deltas[:,0] = tx
    deltas[:,1] = ty
    deltas[:,2] = tw
    deltas[:,3] = th
    
    # set background to -1e8
    idx= gt_boxes[:,0] == -1
    deltas[idx,0] = -1e8
    deltas[idx,1] = -1e8
    deltas[idx,2] = -1e8
    deltas[idx,3] = -1e8
    return deltas


def rcnn_apply_deltas_to_anchors(
    deltas: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """
    Implement the inverse of `rcnn_get_deltas_from_anchors` here.

    Args:
        deltas: Tensor of shape `(N, 4)` giving box regression deltas.
        anchors: Tensor of shape `(N, 4)` giving anchors to apply deltas on.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving the resulting boxes in
            XYXY format.
    """

    # Clamp dw and dh such that they would transform a 8px box no larger than
    # 224px. This is necessary for numerical stability as we apply exponential.
    scale_clamp = math.log(224 / 8)
    deltas[:, 2] = torch.clamp(deltas[:, 2], max=scale_clamp)
    deltas[:, 3] = torch.clamp(deltas[:, 3], max=scale_clamp)

    output_boxes = torch.zeros_like(anchors)
    # delta
    tx = deltas[:, 0]
    ty = deltas[:, 1]
    tw = deltas[:, 2]
    th = deltas[:, 3]
    
    anchors_x1 = anchors[:,0]
    anchors_y1 = anchors[:,1]
    anchors_x2 = anchors[:,2]
    anchors_y2 = anchors[:,3]
    
    pw = anchors_x2 - anchors_x1
    ph = anchors_y2 - anchors_y1
    
    px = (pw) / 2 + anchors_x1
    py = (ph) / 2 + anchors_y1
    
    bx = px + pw * tx
    by = py + ph * ty
    bw = pw * torch.exp(tw)
    bh = ph * torch.exp(th)
    
    x1 = bx - bw/2
    y1 = by - bh/2
    x2 = bx + bw/2
    y2 = by + bh/2
    
    output_boxes[:, 0] = x1
    output_boxes[:, 1] = y1
    output_boxes[:, 2] = x2
    output_boxes[:, 3] = y2
    
    return output_boxes


@torch.no_grad()
def sample_rpn_training(
    gt_boxes: torch.Tensor, num_samples: int, fg_fraction: float
):
    """
    Return `num_samples` (or fewer, if not enough found) random pairs of anchors
    and GT boxes without exceeding `fg_fraction * num_samples` positives, and
    then try to fill the remaining slots with background anchors. We will ignore
    "neutral" anchors in this sampling as they are not used for training.

    Args:
        gt_boxes: Tensor of shape `(N, 5)` giving GT box co-ordinates that are
            already matched with some anchor boxes (with GT class label at last
            dimension). Label -1 means background and -1e8 means meutral.
        num_samples: Total anchor-GT pairs with label >= -1 to return.
        fg_fraction: The number of subsampled labels with values >= 0 is
            `min(num_foreground, int(fg_fraction * num_samples))`. In other
            words, if there are not enough fg, the sample is filled with
            (duplicate) bg.

    Returns:
        fg_idx, bg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or
            fewer. Use these to index anchors, GT boxes, and model predictions.
    """
    foreground = (gt_boxes[:, 4] >= 0).nonzero().squeeze(1)
    background = (gt_boxes[:, 4] == -1).nonzero().squeeze(1)

    # Protect against not enough foreground examples.
    num_fg = min(int(num_samples * fg_fraction), foreground.numel())
    num_bg = num_samples - num_fg

    # Randomly select positive and negative examples.
    perm1 = torch.randperm(foreground.numel(), device=foreground.device)[:num_fg]
    perm2 = torch.randperm(background.numel(), device=background.device)[:num_bg]

    fg_idx = foreground[perm1]
    bg_idx = background[perm2]
    return fg_idx, bg_idx


@torch.no_grad()
def reassign_proposals_to_fpn_levels(
    proposals_per_image: List[torch.Tensor],
    gt_boxes: Optional[torch.Tensor] = None,
    fpn_level_ids: List[int] = [3, 4, 5],
) -> Dict[str, List[torch.Tensor]]:
    """
    The first-stage in Faster R-CNN (RPN) gives a few proposals that are likely
    to contain any object. These proposals would have come from any FPN level -
    for example, they all maybe from level p5, and none from levels p3/p4 (= the
    image mostly has large objects and no small objects). In second stage, these
    proposals are used to extract image features (via RoI-align) and predict the
    class labels. But we do not know which level to use, due to two reasons:

        1. We did not keep track of which level each proposal came from.
        2. ... even if we did keep track, it may be possible that RPN deltas
           transformed a large anchor box from p5 to a tiny proposal (which could
           be more suitable for a lower FPN level).

    Hence, we re-assign proposals to different FPN levels according to sizes.
    Large proposals get assigned to higher FPN levels, and vice-versa.

    At start of training, RPN proposals may be low quality. It's possible that
    very few of these have high IoU with GT boxes. This may stall or de-stabilize
    training of second stage. This function also mixes GT boxes with RPN proposals
    to improve training. GT boxes are also assigned by their size.

    See Equation (1) in FPN paper (https://arxiv.org/abs/1612.03144).

    Args:
        proposals_per_image: List of proposals per image in batch. Same as the
            outputs from `RPN.forward()` method.
        gt_boxes: Tensor of shape `(B, M, 4 or 5)` giving GT boxes per image in
            batch (with or without GT class label, doesn't matter). These are
            not present during inference.
        fpn_levels: List of FPN level IDs. For this codebase this will always
            be [3, 4, 5] for levels (p3, p4, p5) -- we include this in input
            arguments to avoid any hard-coding in function body.

    Returns:
        Dict[str, List[torch.Tensor]]
            Dictionary with keys `{"p3", "p4", "p5"}` each containing a list
            of `B` (`batch_size`) tensors. The `i-th` element in this list will
            give proposals of `i-th` image, assigned to that FPN level. An image
            may not have any proposals for a particular FPN level, for which the
            tensor will be a tensor of shape `(0, 4)` -- PyTorch supports this!
    """

    # Make empty lists per FPN level to add assigned proposals for every image.
    proposals_per_fpn_level = {f"p{_id}": [] for _id in fpn_level_ids}

    # Usually 3 and 5.
    lowest_level_id, highest_level_id = min(fpn_level_ids), max(fpn_level_ids)

    for idx, _props in enumerate(proposals_per_image):

        # Mix ground-truth boxes for every example, per FPN level.
        if gt_boxes is not None:
            # Filter empty GT boxes and remove class label.
            _gtb = gt_boxes[idx]
            _props = torch.cat([_props, _gtb[_gtb[:, 4] != -1][:, :4]], dim=0)

        # Compute FPN level assignments for each GT box. This follows Equation (1)
        # of FPN paper (k0 = 4). `level_assn` has `(M, )` integers, one of {3,4,5}
        _areas = (_props[:, 2] - _props[:, 0]) * (_props[:, 3] - _props[:, 1])

        # Assigned FPN level ID for each proposal (an integer between lowest_level
        # and highest_level).
        level_assignments = torch.floor(4 + torch.log2(torch.sqrt(_areas) / 224))
        level_assignments = torch.clamp(
            level_assignments, min=lowest_level_id, max=highest_level_id
        )
        level_assignments = level_assignments.to(torch.int64)

        # Iterate over FPN level IDs and get proposals for each image, that are
        # assigned to that level.
        for _id in range(lowest_level_id, highest_level_id + 1):
            proposals_per_fpn_level[f"p{_id}"].append(
                # This tensor may have zero proposals, and that's okay.
                _props[level_assignments == _id]
            )

    return proposals_per_fpn_level


class RPN(nn.Module):
    """
    Region Proposal Network: First stage of Faster R-CNN detector.

    This class puts together everything you implemented so far. It accepts FPN
    features as input and uses `RPNPredictionNetwork` to predict objectness and
    box reg deltas. Computes proposal boxes for second stage (during both
    training and inference) and losses during training.
    """

    def __init__(
        self,
        fpn_channels: int,
        stem_channels: List[int],
        batch_size_per_image: int,
        anchor_stride_scale: int = 8,
        anchor_aspect_ratios: List[int] = [0.5, 1.0, 2.0],
        anchor_iou_thresholds: Tuple[int, int] = (0.3, 0.6),
        nms_thresh: float = 0.7,
        pre_nms_topk: int = 400,
        post_nms_topk: int = 100,
    ):
        """
        Args:
            batch_size_per_image: Anchors per image to sample for training.
            nms_thresh: IoU threshold for NMS - unlike FCOS, this is used
                during both, training and inference.
            pre_nms_topk: Number of top-K proposals to select before applying
                NMS, per FPN level. This helps in speeding up NMS.
            post_nms_topk: Number of top-K proposals to select after applying
                NMS, per FPN level. NMS is obviously going to be class-agnostic.

        Refer explanations of remaining args in the classes/functions above.
        """
        super().__init__()
        self.pred_net = RPNPredictionNetwork(
            fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios)
        )
        # Record all input arguments:
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(
        self,
        feats_per_fpn_level: TensorDict,
        strides_per_fpn_level: TensorDict,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        pred_obj_logits, pred_boxreg_deltas = self.pred_net(feats_per_fpn_level)
        
        fpn_feats_shapes = {}
        for level_name, feat in feats_per_fpn_level.items():
            fpn_feats_shapes[level_name] = feat.shape
        
        locations = get_fpn_location_coords(fpn_feats_shapes, strides_per_fpn_level, device=feats_per_fpn_level["p3"].device)
        
        anchors_per_fpn_level = generate_fpn_anchors(locations, 
                                                     strides_per_fpn_level, 
                                                     self.anchor_stride_scale, 
                                                     self.anchor_aspect_ratios)

        # fill three values in this output dict - "proposals",
        # "loss_rpn_box" (training only), "loss_rpn_obj" (training only)
        output_dict = {}

        # Get image height and width according to feature sizes and strides.
        # We need these to clamp proposals (These should be (224, 224) but we
        # avoid hard-coding them).
        img_h = feats_per_fpn_level["p3"].shape[2] * strides_per_fpn_level["p3"]
        img_w = feats_per_fpn_level["p3"].shape[3] * strides_per_fpn_level["p3"]

        output_dict["proposals"] = self.predict_proposals(
            anchors_per_fpn_level,
            pred_obj_logits,
            pred_boxreg_deltas,
            (img_w, img_h),
        )
        # Return here during inference - loss computation not required.
        if not self.training:
            return output_dict

        # Combine anchor boxes from all FPN levels
        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)

        # Get matched GT boxes (list of B tensors, each of shape `(H*W*A, 5)`
        # giving matching GT boxes to anchor boxes).
        matched_gt_boxes = []
        for i in range(num_images):
            matched_boxes_per_fpn_level = rcnn_match_anchors_to_gt(anchor_boxes, gt_boxes[i], self.anchor_iou_thresholds)
            matched_gt_boxes.append(matched_boxes_per_fpn_level)

        # Combine matched boxes from all images to a `(B, HWA, 5)` tensor.
        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)

        # Combine predictions across all FPN levels.
        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)

        if self.training:
            # Repeat anchor boxes `batch_size` times so there is a 1:1
            # correspondence with GT boxes.
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)

            # Collapse `batch_size`, and `HWA` to a single dimension so we have
            # simple `(-1, 4 or 5)` tensors. This simplifies loss computation.
            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)

            loss_obj, loss_box = None, None
            # step 1
            fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes, self.batch_size_per_image * num_images, fg_fraction=0.5)
            # step 2
            gt_deltas_fg = rcnn_get_deltas_from_anchors(anchor_boxes[fg_idx], matched_gt_boxes[fg_idx])
            # step 3
            gt_obj_fg = torch.ones(fg_idx.shape[0], device=matched_gt_boxes.device)
            gt_obj_bg = torch.zeros(bg_idx.shape[0], device=matched_gt_boxes.device)
            
            loss_obj_fg = F.binary_cross_entropy_with_logits(pred_obj_logits[fg_idx], gt_obj_fg, reduction="none")
            loss_obj_bg = F.binary_cross_entropy_with_logits(pred_obj_logits[bg_idx], gt_obj_bg, reduction="none")
            loss_obj = torch.cat((loss_obj_fg, loss_obj_bg), dim=0)
            
            loss_box = F.l1_loss(pred_boxreg_deltas[fg_idx], gt_deltas_fg, reduction="none")
            loss_box[gt_deltas_fg==-1e8] *= 0.0

            # Sum losses and average by num(foreground + background) anchors.
            # In training code, we simply add these two and call `.backward()`
            total_batch_size = self.batch_size_per_image * num_images
            output_dict["loss_rpn_obj"] = loss_obj.sum() / total_batch_size
            output_dict["loss_rpn_box"] = loss_box.sum() / total_batch_size

        return output_dict

    @torch.no_grad()  # Don't track gradients in this function.
    def predict_proposals(
        self,
        anchors_per_fpn_level: Dict[str, torch.Tensor],
        pred_obj_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],  # (width, height)
    ) -> List[torch.Tensor]:
        """
        Predict proposals for a batch of images for the second stage. Other
        input arguments are same as those computed in `forward` method. This
        method should not be called from anywhere except from inside `forward`.

        Returns:
            List[torch.Tensor]
                proposals_per_image: List of B (`batch_size`) tensors givine RPN
                proposals per image. These are boxes in XYXY format, that are
                most likely to contain *any* object. Each tensor in the list has
                shape `(N, 4)` where N could be variable for each image (maximum
                value `post_nms_topk`). These will be anchors for second stage.
        """

        # Gather RPN proposals *from all FPN levels* per image. This will be a
        # list of B (batch_size) tensors giving `(N, 4)` proposal boxes in XYXY
        # format (maximum value of N should be `post_nms_topk`).
        proposals_per_image = []

        # Get batch size to iterate over:
        batch_size = pred_obj_logits["p3"].shape[0]

        for _batch_idx in range(batch_size):

            # For each image in batch, iterate over FPN levels. Fill proposals
            # AND scores per image, per FPN level, in these:
            proposals_per_fpn_level_per_image = {
                level_name: None for level_name in anchors_per_fpn_level.keys()
            }
            scores_per_fpn_level_per_image = {
                level_name: None for level_name in anchors_per_fpn_level.keys()
            }

            for level_name in anchors_per_fpn_level.keys():

                # Get anchor boxes and predictions from a single level.
                level_anchors = anchors_per_fpn_level[level_name]

                # Predictions for a single image - shape: (HWA, ), (HWA, 4)
                level_obj_logits = pred_obj_logits[level_name][_batch_idx]
                level_boxreg_deltas = pred_boxreg_deltas[level_name][_batch_idx]

                proposals_per_image_perB = rcnn_apply_deltas_to_anchors(level_boxreg_deltas, level_anchors)
                w = image_size[0]
                h = image_size[1]
                proposals_per_image_perB[:, 0] = torch.clamp(proposals_per_image_perB[:, 0], min=0, max=w)
                proposals_per_image_perB[:, 1] = torch.clamp(proposals_per_image_perB[:, 1], min=0, max=h)
                proposals_per_image_perB[:, 2] = torch.clamp(proposals_per_image_perB[:, 2], min=0, max=w)
                proposals_per_image_perB[:, 3] = torch.clamp(proposals_per_image_perB[:, 3], min=0, max=h)

                # step 2
                levl_obj_logits = level_obj_logits.reshape(-1)
                if(levl_obj_logits.shape[0] >= self.pre_nms_topk):
                    idx = torch.topk(levl_obj_logits, self.pre_nms_topk).indices
                    proposals_per_image_perB = proposals_per_image_perB[idx]
                    levl_obj_logits = levl_obj_logits[idx]
                else:
                    idx = torch.topk(levl_obj_logits, levl_obj_logits.shape[0]).indices
                    proposals_per_image_perB = proposals_per_image_perB[idx]
                    levl_obj_logits = levl_obj_logits[idx]
                # step 3
                idx = torchvision.ops.nms(proposals_per_image_perB, levl_obj_logits, self.nms_thresh)
                idx = idx[:self.post_nms_topk]
                
                proposals_per_image_perB = proposals_per_image_perB[idx]
                levl_obj_logits = levl_obj_logits[idx]
                
                proposals_per_fpn_level_per_image[level_name] = proposals_per_image_perB
                scores_per_fpn_level_per_image[level_name] = levl_obj_logits

            # Take `post_nms_topk` proposals from all levels combined.
            proposals_all_levels_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            scores_all_levels_per_image = self._cat_across_fpn_levels(
                scores_per_fpn_level_per_image, dim=0
            )
            # Sort scores from highest to smallest and filter proposals.
            _inds = scores_all_levels_per_image.argsort(descending=True)
            _inds = _inds[: self.post_nms_topk]
            keep_proposals = proposals_all_levels_per_image[_inds]

            proposals_per_image.append(keep_proposals)

        return proposals_per_image

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)


class FasterRCNN(nn.Module):
    """
    Faster R-CNN detector: this module combines backbone, RPN, ROI predictors.

    Unlike Faster R-CNN, we will use class-agnostic box regression and Focal
    Loss for classification. We opted for this design choice for you to re-use
    a lot of concepts that you already implemented in FCOS - choosing one loss
    over other matters less overall.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        stem_channels: List[int],
        num_classes: int,
        batch_size_per_image: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.batch_size_per_image = batch_size_per_image

        # Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules using `stem_channels` argument
        cls_pred = []

        conv = nn.Conv2d(self.backbone.out_channels, stem_channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.normal_(conv.weight, mean=0, std=0.01)
        nn.init.zeros_(conv.bias)
        cls_pred.append(conv)
        cls_pred.append(nn.ReLU())
        
        l = len(stem_channels)-1
        for i in range(l):
            conv = nn.Conv2d(stem_channels[i], stem_channels[i+1], kernel_size=3, stride=1, padding=1, bias=True)
            nn.init.normal_(conv.weight, mean=0, std=0.01)
            nn.init.zeros_(conv.bias)
            cls_pred.append(conv)
            cls_pred.append(nn.ReLU())

        cls_pred.append(nn.Flatten())
        cls_pred.append(nn.Linear(self.roi_size[0] * self.roi_size[1] * stem_channels[-1], num_classes + 1))

        # Wrap the layers into a `nn.Sequential` module
        self.cls_pred = nn.Sequential(*cls_pred)

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        See documentation of `FCOS.forward` for more details.
        """

        feats_per_fpn_level = self.backbone(images)
        output_dict = self.rpn(
            feats_per_fpn_level, self.backbone.fpn_strides, gt_boxes
        )
        # List of B (`batch_size`) tensors giving RPN proposals per image.
        proposals_per_image = output_dict["proposals"]

        # Assign the proposals to different FPN levels for extracting features
        # using RoI-align. During training we also mix GT boxes with proposals.
        proposals_per_fpn_level = reassign_proposals_to_fpn_levels(
            proposals_per_image,
            gt_boxes
            # gt_boxes will be None during inference
        )

        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        # Perform RoI-align using FPN features and proposal boxes.
        roi_feats_per_fpn_level = {
            level_name: None for level_name in feats_per_fpn_level.keys()
        }
        # Get RPN proposals from all levels.
        for level_name in feats_per_fpn_level.keys():
            level_feats = feats_per_fpn_level[level_name]
            level_props = proposals_per_fpn_level[level_name]
            level_stride = self.backbone.fpn_strides[level_name]
            roi_feats = torchvision.ops.roi_align(level_feats, 
                                                  level_props,
                                                  output_size=self.roi_size,
                                                  spatial_scale=1/level_stride,
                                                  aligned=True)
            roi_feats_per_fpn_level[level_name] = roi_feats

        # Combine ROI feats across FPN levels, do the same with proposals.
        # shape: (batch_size * total_proposals, fpn_channels, roi_h, roi_w)
        roi_feats = self._cat_across_fpn_levels(roi_feats_per_fpn_level, dim=0)

        # Obtain classification logits for all ROI features.
        # shape: (batch_size * total_proposals, num_classes)
        pred_cls_logits = self.cls_pred(roi_feats)

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass. Batch size must be 1!
            # fmt: off
            return self.inference(
                images,
                proposals_per_fpn_level,
                pred_cls_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on
        # Match the RPN proposals with provided GT boxes and append to `matched_gt_boxes`.
        matched_gt_boxes = []
        for _idx in range(len(gt_boxes)):
            # Get proposals per image from this dictionary of list of tensors.
            proposals_per_fpn_level_per_image = {
                level_name: prop[_idx]
                for level_name, prop in proposals_per_fpn_level.items()
            }

            proposals_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            gt_boxes_per_image = gt_boxes[_idx]
            matched_proposals = rcnn_match_anchors_to_gt(proposals_per_image, gt_boxes_per_image, (0.5, 0.5))
            matched_gt_boxes.append(matched_proposals)

        # Combine predictions and GT from across all FPN levels.
        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

        # Train the classifier head. Perform these steps in order:
        # step 1
        fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes, self.batch_size_per_image * num_images, fg_fraction=0.25)
        gt_cls_fg_idx = matched_gt_boxes[fg_idx]
        gt_cls_bg_idx = matched_gt_boxes[bg_idx]
        
        # step 2
        gt_cls_logit_fg_idx = pred_cls_logits[fg_idx]
        gt_cls_logit_bg_idx = pred_cls_logits[bg_idx]
        gt_labels = matched_gt_boxes[:, 4]
        gt_labels_fg_idx = gt_labels[fg_idx]
        gt_labels_bg_idx = gt_labels[bg_idx]
        
        # step 3
        gt_labels_fg_idx = gt_labels_fg_idx + 1
        gt_labels_bg_idx = gt_labels_bg_idx + 1
        
        gt_labels_comb = torch.cat([gt_labels_fg_idx, gt_labels_bg_idx], dim=0)
        gt_cls_logits_comb = torch.cat([gt_cls_logit_fg_idx, gt_cls_logit_bg_idx], dim=0)
        
        loss_cls = F.cross_entropy(gt_cls_logits_comb, gt_labels_comb.long())
        return {
            "loss_rpn_obj": output_dict["loss_rpn_obj"],
            "loss_rpn_box": output_dict["loss_rpn_box"],
            "loss_cls": loss_cls,
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        proposals: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        test_score_thresh: float,
        test_nms_thresh: float,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions.
        """

        # The second stage inference in Faster R-CNN is quite straightforward:
        # combine proposals from all FPN levels and perform a *class-specific
        # NMS*. There would have been more steps here if we further refined
        # RPN proposals by predicting box regression deltas.

        # Use `[0]` to remove the batch dimension.
        proposals = {level_name: prop[0] for level_name, prop in proposals.items()}
        pred_boxes = self._cat_across_fpn_levels(proposals, dim=0)

        # Faster R-CNN inference
        pred_scores, pred_classes = None, None
        pred_scores_score = torch.max(pred_cls_logits, dim=1).values
        pred_classes = torch.argmax(pred_cls_logits, dim=1) - 1
        
        idx = pred_scores_score > test_score_thresh
        pred_classes = pred_classes[idx]
        pred_scores = pred_scores_score[idx]
        pred_boxes = pred_boxes[idx]

        keep = class_spec_nms(
            pred_boxes, pred_scores, pred_classes, iou_threshold=test_nms_thresh
        )
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        return pred_boxes, pred_classes, pred_scores
