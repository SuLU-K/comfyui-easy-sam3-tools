# comfyui-nodes-test

Small collection of custom nodes used while experimenting with ComfyUI SAM3-based workflows.

## Nodes

| Node | Description |
| ---- | ----------- |
| `SimpleMultiple` | Minimal INT×FLOAT example, kept as a template for quick testing. |
| `Sam3MaskRangeSelector` | Takes the ordered `obj_masks` tensor from SAM3 segmentation, selects a slice (`take_start`, `take_count`) and merges it into a single mask per frame. |
| `Sam3DrawBBox` | Draws KJ-style bounding boxes (from SAM3 or VLM tools) over images, handling stringified JSON, `boxes` payloads, tensor lists, or per-batch structures. |
| `Sam3InteractiveBBoxEditor` | Full interactive canvas node: previews the connected image, lets you draw/move/resize boxes with the mouse, and outputs KJ-format boxes plus the passthrough image. |

### Interactive BBox editor

- Connect any `IMAGE` tensor to the node. The preview auto-resizes like the stock Preview node while keeping coordinates in original resolution.
- Draw boxes with left-click drag, move them by dragging inside, resize via corner handles, delete with the toolbar button or the Delete key.
- The node emits the selected boxes as `[{startX,startY,endX,endY}]`, ready for SAM3 segmentation or `Sam3DrawBBox`.

## Installation

Drop the folder inside `ComfyUI/custom_nodes/`, then restart ComfyUI. The `web/` directory is auto-registered via `WEB_DIRECTORY`, so the interactive widget loads without extra steps.

## Development notes

- The code sticks to ASCII and uses `apply_patch`-friendly formatting.
- All UI work lives under `web/bbox_editor.js`. Update it alongside backend changes and restart ComfyUI’s server + browser to reload.

MIT license applies. Contributions welcome.***
