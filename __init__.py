from .nodes import (
    ConvertPoseDataToPosePoint,
    PosePointSelector,
    Sam3DrawBBox,
    Sam3InteractiveBBoxEditor,
    Sam3MaskRangeSelector,
    SimpleMultiple,
)

NODE_CLASS_MAPPINGS = {
    "SimpleMultiple": SimpleMultiple,
    "Sam3MaskRangeSelector": Sam3MaskRangeSelector,
    "Sam3DrawBBox": Sam3DrawBBox,
    "Sam3InteractiveBBoxEditor": Sam3InteractiveBBoxEditor,
    "ConvertPoseDataToPosePoint": ConvertPoseDataToPosePoint,
    "PosePointSelector": PosePointSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleMultiple": "Simple Multiple",
    "Sam3MaskRangeSelector": "SAM3 Mask Range Selector",
    "Sam3DrawBBox": "SAM3 Draw BBox",
    "Sam3InteractiveBBoxEditor": "SAM3 Interactive BBox",
    "ConvertPoseDataToPosePoint": "Convert PoseData -> PosePoint (OpenPose Editor)",
    "PosePointSelector": "PosePoint Selector (frame)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
