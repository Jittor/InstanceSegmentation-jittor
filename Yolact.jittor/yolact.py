import jittor as jt 
from jittor import nn,init
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict,OrderedDict

from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule,interpolate
from backbone import construct_backbone

from utils import timer
from utils.functions import MovingAverage, make_net


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def execute(self, x):
        # Concat each along the channel dimension
        return jt.contrib.concat([net(x) for net in self.nets], dim=1, **self.extra_params)

prior_cache = defaultdict(lambda: None)

class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim # Defined by Yolact
        self.num_priors  = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index
        self.num_heads   = cfg.num_heads # Defined by Yolact

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            self.mask_dim = self.mask_dim // self.num_heads

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm(out_channels)

            self.bbox_layer = nn.Conv(out_channels, self.num_priors * 4,                **cfg.head_layer_params)
            self.conf_layer = nn.Conv(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)
            
            if cfg.use_mask_scoring:
                self.score_layer = nn.Conv(out_channels, self.num_priors, **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU()
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def execute(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]
        
        conv_h = x.shape[2]
        conv_w = x.shape[3]
        
        if cfg.extra_head_net is not None:
            x = src.upfeature(x)
        
        if cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = nn.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).view(x.shape[0], -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).view(x.shape[0], -1, self.num_classes)
        
        if cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).view(x.shape[0], -1, self.mask_dim)
        else:
            mask = jt.zeros((x.shape[0], bbox.shape[1], self.mask_dim))

        if cfg.use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).view(x.shape[0], -1, 1)

        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).view(x.shape[0], -1, cfg.num_instance_coeffs)    

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = jt.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = jt.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)

                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).view(x.shape[0], -1, self.mask_dim)
                    mask = mask * jt.sigmoid(gate)

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            mask = nn.ConstantPad2d((self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), value=0)(mask)
        
        priors = self.make_priors(conv_h, conv_w)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

        if cfg.use_mask_scoring:
            preds['score'] = score

        if cfg.use_instance_coeff:
            preds['inst'] = inst
        
        return preds

    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        global prior_cache
        size = (conv_h, conv_w)

        with timer.env('makepriors'):
            if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for ars in self.aspect_ratios:
                        for scale in self.scales:
                            for ar in ars:
                                if not cfg.backbone.preapply_sqrt:
                                    ar = sqrt(ar)

                                if cfg.backbone.use_pixel_scales:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                                else:
                                    w = scale * ar / conv_w
                                    h = scale / ar / conv_h
                                
                                # This is for backward compatability with a bug where I made everything square by accident
                                if cfg.backbone.use_square_anchors:
                                    h = w

                                prior_data += [x, y, w, h]

                self.priors = jt.array(prior_data).view(-1, 4)
                self.priors.stop_grad()
                self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
                self.last_conv_size = (conv_w, conv_h)
                prior_cache[size] = None
        
        return self.priors

class FPN(nn.Module):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the execute pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers  = nn.Sequential([
            nn.Conv(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])
        
        self.interpolation_mode     = cfg.fpn.interpolation_mode
        self.num_downsample         = cfg.fpn.num_downsample
        self.use_conv_downsample    = cfg.fpn.use_conv_downsample
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers

    def execute(self, convouts:List[jt.Var]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = jt.zeros((1,))
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers.layers.values():
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].shape
                #print('hh',(h,w),x.shape[-2:])
                x = nn.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
                # x = interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)

            x = x + lat_layer(convouts[j])
            out[j] = x
        
        # This janky second loop is here because jtScript.
        j = len(convouts)
        for pred_layer in self.pred_layers.layers.values():
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                out[j] = nn.relu(out[j])


        cur_idx = len(out)

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers.layers.values():
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks jtScript.
                out.append(nn.pool(out[-1], 1, stride=2,op='maximum'))

        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = nn.relu(out[idx + cur_idx])

        return out

class FastMaskIoUNet(nn.Module):

    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes-1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

    def execute(self, x):
        x = self.maskiou_net(x)
        maskiou_p = nn.pool(x, kernel_size=x.shape[2],op='maximum').squeeze(-1).squeeze(-1)

        return maskiou_p



class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)

        if cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size**2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = jt.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.shape[0]
            else:
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src
            
            if self.proto_src is None: in_channels = 3
            elif cfg.fpn is not None: in_channels = cfg.fpn.num_features
            else: in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1


        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels

        if cfg.use_maskiou:
            self.maskiou_net = FastMaskIoUNet()

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)


        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
                                    scales        = cfg.backbone.pred_scales[idx],
                                    parent        = parent,
                                    index         = idx)
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        
        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv(src_channels[0], cfg.num_classes-1, kernel_size=1)

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
            conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        self.save(path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        import torch
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        # breakpoint()
        self.load_parameters(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        
        
        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pyjt updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    conv_constants = getattr(nn.Conv(1, 1, 1), '__constants__')
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
            
            is_conv_layer = isinstance(module, nn.Conv) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0]  = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.shape[0] - 1)
                        else:
                            module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        jt.nn.init.constant_(module.bias,0.0)
    
    def train(self):
        super().train()
        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pyjt.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm):
                module.train() if enable else module.eval()

                module.weight.stop_grad()
                module.bias.stop_grad()
    
    def execute(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.shape
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        
        with timer.env('backbone'):
            outs = self.backbone(x)

        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)
        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.shape[0], 1, 1, 1)
                    proto_x = jt.contrib.concat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)

                if cfg.mask_proto_prototypes_as_features:
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()

                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                
                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1)

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.shape]
                    bias_shape[-1] = 1
                    proto_out = jt.contrib.concat([proto_out, jt.ones(bias_shape)], -1)

        with timer.env('pred_heads'):
            pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

            if cfg.use_mask_scoring:
                pred_outs['score'] = []

            if cfg.use_instance_coeff:
                pred_outs['inst'] = []
            
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = outs[idx]

                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = nn.interpolate(proto_downsampled, size=outs[idx].shape[2:], mode='bilinear', align_corners=False)
                    # proto_downsampled = interpolate(proto_downsampled, size=outs[idx].shape[2:], mode='bilinear', align_corners=False)
                    
                    pred_x = jt.contrib.concat([pred_x, proto_downsampled], dim=1)

                # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]

                p = pred_layer(pred_x)
                
                for k, v in p.items():
                    # v.sync()
                    pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = jt.contrib.concat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out
        
        #print('hh',pred_outs)
        #print()
        if self.is_training():
            # For the extra loss functions
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            return pred_outs
        else:
            if cfg.use_mask_scoring:
                pred_outs['score'] = jt.sigmoid(pred_outs['score'])
            if cfg.use_focal_loss:
                if cfg.use_sigmoid_focal_loss:
                    # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                    pred_outs['conf'] = jt.sigmoid(pred_outs['conf'])
                    if cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif cfg.use_objectness_score:
                    # See focal_loss_sigmoid in multibox_loss.py for details
                    objectness = jt.sigmoid(pred_outs['conf'][:, :, 0])
                    pred_outs['conf'][:, :, 1:] = objectness.unsqueeze(2) * nn.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, 0 ] = 1 - objectness
                else:
                    pred_outs['conf'] = nn.softmax(pred_outs['conf'], -1)
            else:

                if cfg.use_objectness_score:
                    objectness = jt.sigmoid(pred_outs['conf'][:, :, 0])
                    
                    pred_outs['conf'][:, :, 1:] = (objectness > 0.10).unsqueeze(-1) \
                        * nn.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
                    
                else:
                    pred_outs['conf'] = nn.softmax(pred_outs['conf'], -1)
            return self.detect(pred_outs, self)




# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    # Use the first argument to set the config if you want
    import sys
    if len(sys.argv) > 1:
        from data.config import set_cfg
        set_cfg(sys.argv[1])

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    x = jt.zeros((1, 3, cfg.max_size, cfg.max_size))
    y = net(x)

    for p in net.prediction_layers:
        print(p.last_conv_size)

    print()
    for k, a in y.items():
        print(k + ': ', a.shape, jt.sum(a))
    exit()
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
