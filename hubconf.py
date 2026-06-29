dependencies = ['torch']
from modules.xfeat import XFeat as _XFeat
from modules.xfeat import resolve_inference_device
import torch

def XFeat(pretrained=True, top_k=4096, detection_threshold=0.05):
    """
    XFeat model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    weights = None
    dev = None
    if pretrained:
        dev = resolve_inference_device("auto")
        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt",
            map_location=dev,
        )

    model = _XFeat(weights, top_k=top_k, detection_threshold=detection_threshold, device=dev)
    return model
