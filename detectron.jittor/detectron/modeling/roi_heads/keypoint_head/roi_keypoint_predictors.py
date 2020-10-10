from jittor import nn,Module,init
from detectron import layers
from detectron.modeling import registry


@registry.ROI_KEYPOINT_PREDICTOR.register("KeypointRCNNPredictor")
class KeypointRCNNPredictor(Module):
    def __init__(self, cfg, in_channels):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        deconv_kernel = 4
        self.kps_score_lowres = layers.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def execute(self, x):
        x = self.kps_score_lowres(x)
        x = layers.interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x


def make_roi_keypoint_predictor(cfg, in_channels):
    func = registry.ROI_KEYPOINT_PREDICTOR[cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR]
    return func(cfg, in_channels)
