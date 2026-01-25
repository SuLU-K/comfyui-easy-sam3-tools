import base64
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageDraw


class SimpleMultiple:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_int": ("INT", {"forceInput": True}),
                "multiplier": ("FLOAT", {"default": 2.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result_int",)
    FUNCTION = "compute_multiple"
    CATEGORY = "Test Nodes/Math"

    def compute_multiple(self, input_int, multiplier):
        result = int(input_int * multiplier)
        print(f"Calculation: {input_int} * {multiplier} = {result}")
        return (result,)


class Sam3MaskRangeSelector:
    """Select a range of ordered SAM3 masks and merge them into a single mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "obj_masks": ("MASK", {"forceInput": True}),
                "take_start": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "take_count": ("INT", {"default": 1, "min": -1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select_and_merge"
    CATEGORY = "Test Nodes/SAM3"

    def select_and_merge(self, obj_masks, take_start, take_count):
        masks = self._ensure_tensor(obj_masks)
        batch, num_objects, height, width = masks.shape

        start = max(0, min(take_start, num_objects))
        if take_count < 0:
            count = num_objects - start
        else:
            count = take_count
        end = max(start, min(start + count, num_objects))

        if end == start:
            empty = torch.zeros((batch, height, width), dtype=masks.dtype, device=masks.device)
            return (empty,)

        selected = masks[:, start:end]
        merged = (selected.sum(dim=1) > 0).to(dtype=masks.dtype)
        return (merged,)

    @staticmethod
    def _ensure_tensor(obj_masks):
        if isinstance(obj_masks, torch.Tensor):
            tensor = obj_masks
        elif isinstance(obj_masks, (list, tuple)):
            if len(obj_masks) == 0:
                raise ValueError("obj_masks is empty")

            # Upstream nodes are inconsistent about whether they emit:
            # - a 4D tensor: [B, N, H, W]
            # - a list of per-object 3D tensors: N * [B, H, W]
            # - a list of per-frame 3D arrays: B * [N, H, W]
            #
            # Prefer treating a list of torch tensors as per-object masks, since that
            # is the common shape that can otherwise get misinterpreted as [N, B, H, W].
            if all(isinstance(m, torch.Tensor) for m in obj_masks):
                elems = []
                for m in obj_masks:
                    t = m
                    if t.dim() == 2:
                        t = t.unsqueeze(0)  # [1, H, W]
                    elems.append(t)
                if all(t.dim() == 3 for t in elems) and all(t.shape == elems[0].shape for t in elems):
                    tensor = torch.stack(elems, dim=1)  # [B, N, H, W]
                else:
                    tensor = torch.stack(elems, dim=0)
            else:
                tensor = torch.stack([
                    m if isinstance(m, torch.Tensor) else torch.tensor(m)
                    for m in obj_masks
                ])
        else:
            tensor = torch.tensor(obj_masks)

        if tensor.dim() == 5 and tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)

        if tensor.dim() != 4:
            raise ValueError(f"obj_masks must be a 4D tensor, got shape {tensor.shape}")

        # Normalize common swapped layout [N, B, H, W] -> [B, N, H, W]
        # (often happens when upstream emits a list of per-object [B, H, W] masks).
        if tensor.shape[0] > 1 and tensor.shape[1] == 1:
            tensor = tensor.permute(1, 0, 2, 3).contiguous()

        return tensor.float()


class Sam3DrawBBox:
    """Draw bounding boxes on images. Supports KJ format and raw xyxy/xywh lists."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bboxs": ("BBOX", {"forceInput": True}),
                "stroke_width": ("INT", {"default": 4, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "draw"
    CATEGORY = "Test Nodes/SAM3"

    def draw(self, images, bboxs, stroke_width):
        pil_images = self._tensor_to_pil(images)
        boxes_per_image = self._prepare_boxes(bboxs, len(pil_images))

        drawn = []
        for idx, pil_img in enumerate(pil_images):
            img_boxes = boxes_per_image[idx] if idx < len(boxes_per_image) else boxes_per_image[-1]
            width, height = pil_img.size
            draw = ImageDraw.Draw(pil_img)

            for box in img_boxes:
                coords = self._to_xyxy(box, width, height)
                if coords is None:
                    continue
                x1, y1, x2, y2 = coords
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=stroke_width)

            drawn.append(self._pil_to_tensor(pil_img))

        output = torch.stack(drawn, dim=0)
        return (output,)

    @staticmethod
    def _to_python(obj):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                return None
        return obj

    @classmethod
    def _prepare_boxes(cls, bboxs, batch):
        bboxs = cls._to_python(bboxs)
        if bboxs is None:
            return [[] for _ in range(batch)]

        # Handle dict with "boxes" field
        if isinstance(bboxs, dict) and "boxes" in bboxs:
            bboxs = bboxs["boxes"]

        # Check for per-image list
        if isinstance(bboxs, (list, tuple)) and len(bboxs) == batch and all(cls._is_box_collection(elem) for elem in bboxs):
            return [cls._normalize_box_list(elem) for elem in bboxs]

        normalized = cls._normalize_box_list(bboxs)
        return [normalized for _ in range(batch)]

    @staticmethod
    def _is_box_collection(obj):
        if obj is None:
            return True
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return True
            if len(obj) == 4 and all(isinstance(v, (int, float)) for v in obj):
                return False
            return True
        if isinstance(obj, dict):
            return True
        return False

    @classmethod
    def _normalize_box_list(cls, boxes):
        boxes = cls._to_python(boxes)
        if boxes is None:
            return []

        if isinstance(boxes, dict) and {"startX", "startY", "endX", "endY"} <= boxes.keys():
            return [boxes]

        if isinstance(boxes, dict) and "boxes" in boxes:
            boxes = boxes["boxes"]

        if isinstance(boxes, (list, tuple)):
            if len(boxes) == 0:
                return []
            if len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
                return [boxes]
            normalized = []
            for item in boxes:
                if item is None:
                    continue
                if isinstance(item, dict) and {"startX", "startY", "endX", "endY"} <= item.keys():
                    normalized.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 4:
                    normalized.append([float(v) for v in list(item)[:4]])
            return normalized

        return []

    @staticmethod
    def _to_xyxy(box, width, height):
        if box is None:
            return None

        if isinstance(box, dict):
            if {"startX", "startY", "endX", "endY"} <= box.keys():
                x1 = float(box["startX"])
                y1 = float(box["startY"])
                x2 = float(box["endX"])
                y2 = float(box["endY"])
                return Sam3DrawBBox._clamp_box((x1, y1, x2, y2), width, height)
            if {"x", "y", "w", "h"} <= box.keys():
                x1 = float(box["x"])
                y1 = float(box["y"])
                x2 = x1 + float(box["w"])
                y2 = y1 + float(box["h"])
                return Sam3DrawBBox._clamp_box((x1, y1, x2, y2), width, height)

        if isinstance(box, (torch.Tensor, np.ndarray)):
            box = box.tolist()

        if isinstance(box, (list, tuple)) and len(box) >= 4:
            vals = [float(v) for v in box[:4]]
            x1, y1, x2, y2 = vals
            if x2 <= x1 or y2 <= y1:
                # Treat as xywh (top-left or normalized center)
                if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
                    cx, cy, w, h = vals
                    x1 = (cx - w / 2.0) * width
                    y1 = (cy - h / 2.0) * height
                    x2 = (cx + w / 2.0) * width
                    y2 = (cy + h / 2.0) * height
                else:
                    w, h = x2, y2
                    x2 = x1 + w
                    y2 = y1 + h
            else:
                if max(abs(x1), abs(x2)) <= 1.0 and max(abs(y1), abs(y2)) <= 1.0:
                    x1 *= width
                    x2 *= width
                    y1 *= height
                    y2 *= height
            return Sam3DrawBBox._clamp_box((x1, y1, x2, y2), width, height)

        return None

    @staticmethod
    def _clamp_box(coords, width, height):
        x1, y1, x2, y2 = coords
        x1 = max(0.0, min(width, x1))
        y1 = max(0.0, min(height, y1))
        x2 = max(0.0, min(width, x2))
        y2 = max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _tensor_to_pil(images):
        imgs = images.detach().cpu().numpy()
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        imgs = np.clip(imgs, 0.0, 1.0)
        imgs = (imgs * 255).astype(np.uint8)
        pil_list = []
        for img in imgs:
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                pil_list.append(Image.fromarray(img))
            elif img.ndim == 3 and img.shape[0] in (3, 4):
                pil_list.append(Image.fromarray(np.moveaxis(img, 0, -1)))
            else:
                pil_list.append(Image.fromarray(img.squeeze(), mode="L"))
        return pil_list

    @staticmethod
    def _pil_to_tensor(pil_img):
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)


class Sam3InteractiveBBoxEditor:
    """Interactive editor that previews the image and lets users drag bounding boxes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "bbox_data": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxs", "images")
    FUNCTION = "collect"
    CATEGORY = "Test Nodes/SAM3"

    def collect(self, images, bbox_data=""):
        bbox_list = self._parse_bbox_data(bbox_data, images)
        preview_payload = self._build_preview_payload(images)

        if preview_payload is None:
            return (bbox_list, images)

        return {
            "result": (bbox_list, images),
            "ui": {"bbox_preview": [preview_payload]},
        }

    @staticmethod
    def _parse_bbox_data(raw_data, images):
        if not raw_data:
            return []

        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return []

        boxes = payload.get("boxes", payload if isinstance(payload, list) else [])
        if not isinstance(boxes, list):
            return []

        height = float(images.shape[1])
        width = float(images.shape[2])
        parsed = []

        for box in boxes:
            if not isinstance(box, dict):
                continue

            x = Sam3InteractiveBBoxEditor._get_float(box, "x")
            y = Sam3InteractiveBBoxEditor._get_float(box, "y")
            w = Sam3InteractiveBBoxEditor._get_float(box, "w")
            h = Sam3InteractiveBBoxEditor._get_float(box, "h")

            if w is None or h is None or x is None or y is None:
                start_x = Sam3InteractiveBBoxEditor._get_float(box, "startX")
                end_x = Sam3InteractiveBBoxEditor._get_float(box, "endX")
                start_y = Sam3InteractiveBBoxEditor._get_float(box, "startY")
                end_y = Sam3InteractiveBBoxEditor._get_float(box, "endY")
                if None in (start_x, start_y, end_x, end_y):
                    continue
                x, y = start_x, start_y
                w = end_x - start_x
                h = end_y - start_y

            if w <= 1 or h <= 1:
                continue

            start_x = float(np.clip(x, 0.0, width))
            start_y = float(np.clip(y, 0.0, height))
            end_x = float(np.clip(x + w, 0.0, width))
            end_y = float(np.clip(y + h, 0.0, height))

            if end_x <= start_x or end_y <= start_y:
                continue

            parsed.append(
                {
                    "startX": start_x,
                    "startY": start_y,
                    "endX": end_x,
                    "endY": end_y,
                }
            )

        return parsed

    @staticmethod
    def _build_preview_payload(images):
        if not isinstance(images, torch.Tensor) or images.shape[0] == 0:
            return None

        frame = images[0].detach().cpu().numpy()
        frame = np.clip(frame, 0.0, 1.0)

        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            mode = "RGBA" if frame.shape[-1] == 4 else "RGB"
            arr = (frame * 255).astype(np.uint8)
        elif frame.ndim == 3 and frame.shape[0] in (3, 4):
            arr = np.moveaxis(frame, 0, -1)
            mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = (frame.squeeze() * 255).astype(np.uint8)
            mode = "L"

        pil_image = Image.fromarray(arr, mode=mode)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        height = int(images.shape[1])
        width = int(images.shape[2])
        return {"image": encoded, "width": width, "height": height}

    @staticmethod
    def _get_float(data, key):
        value = data.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _to_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_xy(x, y, width, height):
    if x is None or y is None:
        return None, None

    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return None, None

    if width and height and (x > 1.5 or y > 1.5):
        if width > 0:
            x = x / float(width)
        if height > 0:
            y = y / float(height)

    return x, y


def _meta_to_pose_keypoint(meta):
    if not isinstance(meta, dict):
        return None

    width = _to_int(meta.get("width"), default=512) or 512
    height = _to_int(meta.get("height"), default=512) or 512

    keypoints_body = meta.get("keypoints_body")
    if keypoints_body is None:
        return None

    if isinstance(keypoints_body, np.ndarray):
        points = [row for row in keypoints_body]
    elif isinstance(keypoints_body, list):
        points = keypoints_body
    else:
        return None

    flat = [0.0] * 54  # 18 * (x, y, confidence)
    for i in range(min(18, len(points))):
        kp = points[i]
        if kp is None:
            continue

        if isinstance(kp, np.ndarray):
            kp = kp.tolist()

        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue

        x, y = kp[0], kp[1]
        conf = kp[2] if len(kp) >= 3 else 1.0

        x, y = _normalize_xy(x, y, width, height)
        if x is None or y is None:
            continue

        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0

        base = i * 3
        flat[base] = float(x)
        flat[base + 1] = float(y)
        flat[base + 2] = float(conf)

    return {
        "canvas_width": int(width),
        "canvas_height": int(height),
        "width": int(width),
        "height": int(height),
        "people": [{"pose_keypoints_2d": flat}],
    }


class ConvertPoseDataToPosePoint:
    """Convert PoseAndFaceDetection POSEDATA into OpenPose-Editor POSE_KEYPOINT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_point",)
    FUNCTION = "convert"
    CATEGORY = "Test Nodes/Pose"

    def convert(self, pose_data):
        if not isinstance(pose_data, dict):
            raise ValueError("pose_data must be a dict (POSEDATA)")

        metas = pose_data.get("pose_metas_original")
        if metas is None:
            metas = pose_data.get("pose_metas")

        if not isinstance(metas, list) or len(metas) == 0:
            raise ValueError("pose_data has no pose metas to convert")

        pose_point = []
        for meta in metas:
            converted = _meta_to_pose_keypoint(meta)
            if converted is not None:
                pose_point.append(converted)

        if len(pose_point) == 0:
            raise ValueError("failed to convert pose metas into POSE_KEYPOINT format")

        return (pose_point,)


class PosePointSelector:
    """Select a single frame from a POSE_KEYPOINT sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_point": ("POSE_KEYPOINT", {"forceInput": True}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "INT")
    RETURN_NAMES = ("pose_point", "selected_index")
    FUNCTION = "select"
    CATEGORY = "Test Nodes/Pose"

    def select(self, pose_point, frame_index):
        if not isinstance(pose_point, list) or len(pose_point) == 0:
            raise ValueError("pose_point must be a non-empty list (POSE_KEYPOINT)")

        try:
            idx = int(frame_index)
        except (TypeError, ValueError):
            idx = 0

        idx = max(0, min(idx, len(pose_point) - 1))
        return ([pose_point[idx]], idx)
