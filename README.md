# comfyui-nodes-test

![SAM3 Interactive BBox Editor](assets/image.png)

Custom nodes for building SAM3-centric editing pipelines inside ComfyUI. They pair well with **easy-sam3**, letting you harness SAM3’s zero-shot segmentation accuracy for precise masks that can drive local repainting, matting, or any downstream compositing task.

## Nodes

| Node | Description |
| ---- | ----------- |
| `SimpleMultiple` | Minimal INT×FLOAT demo node. Useful as a boilerplate when authoring new operators. |
| `Sam3MaskRangeSelector` | Consumes ordered SAM3 `obj_masks`, extracts a configurable slice (`take_start`, `take_count`), and merges the selection into a single mask per frame for focused edits. |
| `Sam3DrawBBox` | Renders SAM3 / VLM bounding boxes over image tensors. Accepts JSON strings, `{boxes: [...]}` payloads, tensors, or per-batch lists, normalizes them, then emits an overlayed tensor. |
| `Sam3InteractiveBBoxEditor` | DOM-based bbox editor widget. Streams the upstream image, lets you author/adjust boxes interactively, and outputs KJ-format bounds alongside the passthrough image for subsequent SAM3 inference. |

### Interactive BBox editor

- Connect any `IMAGE` tensor to the node. The preview auto-resizes like the stock Preview node while preserving original-resolution coordinates.
- Draw boxes with left-click drag, move them by dragging inside, resize via corner handles, delete with the toolbar button or the Delete key.
- The node emits the selected boxes as `[{startX,startY,endX,endY}]`, ready for SAM3 segmentation or `Sam3DrawBBox`.

## Installation

Drop the folder inside `ComfyUI/custom_nodes/`, then restart ComfyUI.

MIT license applies. Contributions welcome.
