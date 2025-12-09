import { app } from "../../scripts/app.js";

const HANDLE_SIZE = 8;
const MIN_SIZE = 6;

function hideWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function toCanvasCoords(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
    };
}

function normalizeBoxes(list) {
    if (!Array.isArray(list)) return [];
    const normalized = [];
    list.forEach((item) => {
        if (!item || typeof item !== "object") {
            return;
        }
        let x = Number(item.x);
        let y = Number(item.y);
        let w = Number(item.w);
        let h = Number(item.h);

        if ([x, y, w, h].some((v) => Number.isNaN(v))) {
            const sx = Number(item.startX);
            const sy = Number(item.startY);
            const ex = Number(item.endX);
            const ey = Number(item.endY);
            if (![sx, sy, ex, ey].some((v) => Number.isNaN(v))) {
                x = sx;
                y = sy;
                w = ex - sx;
                h = ey - sy;
            }
        }

        if ([x, y, w, h].some((v) => Number.isNaN(v))) {
            return;
        }

        if (w <= 0 || h <= 0) {
            return;
        }

        normalized.push({ x, y, w, h });
    });
    return normalized;
}

function parseBoxes(value) {
    if (!value) return [];
    try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed)) {
            return normalizeBoxes(parsed);
        }
        if (parsed && Array.isArray(parsed.boxes)) {
            return normalizeBoxes(parsed.boxes);
        }
    } catch (err) {
        console.warn("[InteractiveBBox] Failed to parse bbox data:", err);
    }
    return [];
}

function serializeBoxes(boxes) {
    return JSON.stringify({
        boxes: boxes.map((box) => ({
            x: box.x,
            y: box.y,
            w: box.w,
            h: box.h,
        })),
    });
}

function createHandleRects(box) {
    const { x, y, w, h } = box;
    const right = x + w;
    const bottom = y + h;
    const size = HANDLE_SIZE;
    return [
        { corner: "nw", x: x - size, y: y - size },
        { corner: "ne", x: right - size, y: y - size },
        { corner: "sw", x: x - size, y: bottom - size },
        { corner: "se", x: right - size, y: bottom - size },
    ];
}

function pointInRect(px, py, rect) {
    return px >= rect.x && px <= rect.x + HANDLE_SIZE * 2 && py >= rect.y && py <= rect.y + HANDLE_SIZE * 2;
}

function hitTestHandles(boxes, px, py) {
    for (let i = boxes.length - 1; i >= 0; i -= 1) {
        const handles = createHandleRects(boxes[i]);
        for (const handle of handles) {
            if (pointInRect(px, py, handle)) {
                return { index: i, corner: handle.corner };
            }
        }
    }
    return null;
}

function hitTestBoxes(boxes, px, py) {
    for (let i = boxes.length - 1; i >= 0; i -= 1) {
        const box = boxes[i];
        if (px >= box.x && px <= box.x + box.w && py >= box.y && py <= box.y + box.h) {
            return i;
        }
    }
    return -1;
}

function attachEditor(node, widget) {
    const container = document.createElement("div");
    container.style.cssText = "position: relative; width: 100%; background: #111; border-radius: 6px; border: 1px solid #222; overflow: hidden; display: flex; flex-direction: column; gap: 4px;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: #1b1c1f; border-bottom: 1px solid #222; font-size: 12px; color: #bbb;";
    toolbar.innerHTML = "<span>Click & drag to draw a box. Drag edges/handles to adjust.</span>";

    const toolbarButtons = document.createElement("div");
    toolbarButtons.style.cssText = "display: flex; gap: 6px;";

    const createButton = (label, handler) => {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.style.cssText = "background: #2f3034; color: #eee; border: 1px solid #3c3d42; border-radius: 4px; padding: 2px 8px; cursor: pointer;";
        btn.onmouseenter = () => (btn.style.background = "#3b3c41");
        btn.onmouseleave = () => (btn.style.background = "#2f3034");
        btn.onclick = handler;
        return btn;
    };

    const clearBtn = createButton("Clear", () => {
        state.boxes = [];
        state.selected = -1;
        state.action = null;
        state.save();
        state.draw();
    });

    const deleteBtn = createButton("Delete", () => {
        if (state.selected === -1) {
            return;
        }
        state.boxes.splice(state.selected, 1);
        state.selected = -1;
        state.save();
        state.draw();
    });

    toolbarButtons.appendChild(clearBtn);
    toolbarButtons.appendChild(deleteBtn);
    toolbar.appendChild(toolbarButtons);
    container.appendChild(toolbar);

    const canvasWrapper = document.createElement("div");
    canvasWrapper.style.cssText = "flex: 1; display: flex; align-items: center; justify-content: center; background: #0d0e10; padding: 6px;";

    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    canvas.style.cssText = "max-width: 100%; max-height: 100%; cursor: crosshair; background: #0d0e10;";
    canvasWrapper.appendChild(canvas);
    container.appendChild(canvasWrapper);

    const ctx = canvas.getContext("2d");

    const state = {
        canvas,
        ctx,
        widget,
        boxes: parseBoxes(widget?.value),
        selected: -1,
        action: null,
        image: null,
        previewSize: { width: canvas.width, height: canvas.height },
        originalSize: { width: canvas.width, height: canvas.height },
        scale: 1,
        toDisplayBox(box) {
            const scale = this.scale || 1;
            return {
                x: box.x * scale,
                y: box.y * scale,
                w: box.w * scale,
                h: box.h * scale,
            };
        },
        toDisplayCoords(coords) {
            const scale = this.scale || 1;
            return {
                x: coords.x * scale,
                y: coords.y * scale,
            };
        },
        toActualCoords(coords) {
            const scale = this.scale || 1;
            const inv = scale === 0 ? 1 : 1 / scale;
            return {
                x: coords.x * inv,
                y: coords.y * inv,
            };
        },
        toActualBox(box) {
            const scale = this.scale || 1;
            const inv = scale === 0 ? 1 : 1 / scale;
            return {
                x: box.x * inv,
                y: box.y * inv,
                w: box.w * inv,
                h: box.h * inv,
            };
        },
        save() {
            if (this.widget) {
                this.widget.value = serializeBoxes(this.boxes);
            }
            app.graph.setDirtyCanvas(true);
        },
        setImageFromMessage(message) {
            if (!message?.bbox_preview || !message.bbox_preview[0]) {
                return;
            }
            const payload = message.bbox_preview[0];
            const base64 = typeof payload === "string" ? payload : payload.image;
            if (!base64) {
                return;
            }
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this.originalSize = { width: img.width, height: img.height };
                const maxDim = 640;
                const longest = Math.max(img.width, img.height);
                const scale = longest > maxDim ? maxDim / longest : 1;
                const displayWidth = Math.max(64, Math.round(img.width * scale));
                const displayHeight = Math.max(64, Math.round(img.height * scale));
                this.scale = scale;
                this.previewSize = { width: displayWidth, height: displayHeight };
                this.canvas.width = displayWidth;
                this.canvas.height = displayHeight;
                this.draw();
            };
            img.src = `data:image/png;base64,${base64}`;
        },
        draw() {
            const { canvas, ctx, boxes, selected, image } = this;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (image) {
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            } else {
                ctx.fillStyle = "#15161a";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#5c5d63";
                ctx.font = "16px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Connect an image input to start drawing boxes", canvas.width / 2, canvas.height / 2);
            }

            const displayBoxes = boxes.map((box) => this.toDisplayBox(box));

            displayBoxes.forEach((box, index) => {
                const color = index === selected ? "#ff9800" : "#3ab0ff";
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(box.x, box.y, box.w, box.h);

                if (index === selected) {
                    ctx.fillStyle = "rgba(255, 152, 0, 0.15)";
                    ctx.fillRect(box.x, box.y, box.w, box.h);
                    ctx.fillStyle = "#fff";
                    const handles = createHandleRects(box);
                    handles.forEach((handle) => {
                        ctx.fillRect(handle.x, handle.y, HANDLE_SIZE * 2, HANDLE_SIZE * 2);
                    });
                }
            });

            if (this.action && this.action.type === "draw" && this.action.current) {
                ctx.strokeStyle = "#ff9800";
                ctx.setLineDash([6, 4]);
                const currentDisplay = this.toDisplayBox(this.action.current);
                const b = currentDisplay;
                ctx.strokeRect(b.x, b.y, b.w, b.h);
                ctx.setLineDash([]);
            }
        },
    };

    node.sam3Interactive = state;

    const domWidget = node.addDOMWidget("bbox", "sam3-bbox", container);
    domWidget.computeSize = (width) => {
        const nodeHeight = node.size ? node.size[1] : 460;
        const height = Math.max(260, nodeHeight - 140);
        container.style.height = `${height}px`;
        return [width, height];
    };

    const originalResize = node.onResize;
    node.onResize = function (size) {
        originalResize?.apply(this, arguments);
        const height = Math.max(260, size[1] - 140);
        container.style.height = `${height}px`;
    };

    const startAction = (action) => {
        state.action = action;
    };

    const finishAction = () => {
        if (!state.action) return;
        if (state.action.type === "draw" && state.action.current) {
            const box = state.action.current;
            if (box.w > MIN_SIZE && box.h > MIN_SIZE) {
                state.boxes.push({ ...box });
                state.selected = state.boxes.length - 1;
                state.save();
            }
        } else if (state.action.type === "move" || state.action.type === "resize") {
            state.save();
        }
        state.action = null;
        state.draw();
    };

    const handleMouseDown = (event) => {
        event.preventDefault();
        const { canvas, boxes } = state;
        const coords = toCanvasCoords(canvas, event);
        const displayBoxes = boxes.map((box) => state.toDisplayBox(box));
        const handleHit = hitTestHandles(displayBoxes, coords.x, coords.y);
        if (handleHit) {
            const target = boxes[handleHit.index];
            state.selected = handleHit.index;
            const actualCoords = state.toActualCoords(coords);
            startAction({
                type: "resize",
                index: handleHit.index,
                corner: handleHit.corner,
                start: actualCoords,
                startBox: { ...target },
            });
            state.draw();
            return;
        }

        const boxIndex = hitTestBoxes(displayBoxes, coords.x, coords.y);
        if (boxIndex !== -1) {
            state.selected = boxIndex;
            const target = boxes[boxIndex];
            const actualCoords = state.toActualCoords(coords);
            startAction({
                type: "move",
                index: boxIndex,
                start: actualCoords,
                offset: { x: actualCoords.x - target.x, y: actualCoords.y - target.y },
                startBox: { ...target },
            });
            state.draw();
            return;
        }

        const actualCoords = state.toActualCoords(coords);
        startAction({
            type: "draw",
            start: actualCoords,
            current: { x: actualCoords.x, y: actualCoords.y, w: 0, h: 0 },
        });
        state.draw();
    };

    const handleMouseMove = (event) => {
        if (!state.action) return;
        const { canvas, boxes, action } = state;
        const coords = toCanvasCoords(canvas, event);

        if (action.type === "draw") {
            const actualCoords = state.toActualCoords(coords);
            const x = Math.min(action.start.x, actualCoords.x);
            const y = Math.min(action.start.y, actualCoords.y);
            const w = Math.abs(actualCoords.x - action.start.x);
            const h = Math.abs(actualCoords.y - action.start.y);
            action.current = { x, y, w, h };
            state.draw();
            return;
        }

        if (action.type === "move") {
            const box = boxes[action.index];
            const actualCoords = state.toActualCoords(coords);
            const bounds = state.originalSize;
            const nextX = actualCoords.x - action.offset.x;
            const nextY = actualCoords.y - action.offset.y;
            box.x = clamp(nextX, 0, bounds.width - box.w);
            box.y = clamp(nextY, 0, bounds.height - box.h);
            state.draw();
            return;
        }

        if (action.type === "resize") {
            const box = boxes[action.index];
            const startBox = action.startBox;
            let { x, y, w, h } = startBox;
            const maxX = state.originalSize.width;
            const maxY = state.originalSize.height;
            const actualCoords = state.toActualCoords(coords);
            const minActual = MIN_SIZE / (state.scale || 1);

            if (action.corner === "nw") {
                const newX = clamp(actualCoords.x, 0, x + w - minActual);
                const newY = clamp(actualCoords.y, 0, y + h - minActual);
                w = w + (x - newX);
                h = h + (y - newY);
                x = newX;
                y = newY;
            } else if (action.corner === "ne") {
                const newX = clamp(actualCoords.x, x + minActual, maxX);
                const newY = clamp(actualCoords.y, 0, y + h - minActual);
                w = newX - x;
                h = h + (y - newY);
                y = newY;
            } else if (action.corner === "sw") {
                const newX = clamp(actualCoords.x, 0, x + w - minActual);
                const newY = clamp(actualCoords.y, y + minActual, maxY);
                w = w + (x - newX);
                h = newY - y;
                x = newX;
            } else if (action.corner === "se") {
                const newX = clamp(actualCoords.x, x + minActual, maxX);
                const newY = clamp(actualCoords.y, y + minActual, maxY);
                w = newX - x;
                h = newY - y;
            }

            box.x = x;
            box.y = y;
            box.w = Math.max(minActual, Math.min(w, maxX - x));
            box.h = Math.max(minActual, Math.min(h, maxY - y));
            state.draw();
        }
    };

    const handleMouseUp = () => {
        if (!state.action) return;
        finishAction();
    };

    canvas.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    window.addEventListener("keydown", (event) => {
        if ((event.key === "Delete" || event.key === "Backspace") && state.selected !== -1) {
            state.boxes.splice(state.selected, 1);
            state.selected = -1;
            state.save();
            state.draw();
        }
    });

    state.draw();
}

app.registerExtension({
    name: "comfyui-nodes-test.interactive-bbox",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const targetNames = ["Sam3InteractiveBBoxEditor", "SAM3 Interactive BBox"];
        if (!targetNames.includes(nodeData.name)) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            const widget = this.widgets?.find((w) => w.name === "bbox_data");
            if (widget) {
                hideWidget(widget);
            }
            attachEditor(this, widget);
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalOnExecuted?.apply(this, arguments);
            this.sam3Interactive?.setImageFromMessage(message);
        };
    },
});
