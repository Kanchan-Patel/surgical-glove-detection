import argparse
import json
import threading
import traceback
from pathlib import Path
from typing import List

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk


# ---------------- Utilities ----------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def draw_box(img, xyxy, label, conf, thickness=2):
    x1, y1, x2, y2 = map(int, map(round, xyxy))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(img, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def to_scalar(x):
    try:
        return float(x.item()) if hasattr(x, "item") else float(x)
    except Exception:
        return float(x)


# ---------------- Detection Logic ----------------

class Detector:
    def __init__(self, model_path: str, device: str = None, imgsz: int = 640, conf: float = 0.5, iou: float = 0.45):
        self.model = None
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def load(self):
        if self.model is None:
            try:
                self.model = YOLO(self.model_path)
                if self.device:
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
            except Exception as e:
                raise RuntimeError(f"Failed to load model '{self.model_path}': {e}")

    def predict(self, img: np.ndarray):
        self.load()
        result = self.model.predict(
            img, imgsz=self.imgsz, conf=self.conf,
            iou=self.iou, device=self.device, verbose=False
        )[0]

        names = result.names
        if isinstance(names, dict):
            maxk = max(names.keys()) if len(names) else -1
            names = [names.get(i, str(i)) for i in range(maxk + 1)]

        detections = []
        annotated = img.copy()

        for box in getattr(result, "boxes", []):
            try:
                xy = box.xyxy
                xyxy = xy.cpu().numpy().ravel().tolist()
            except Exception:
                continue

            try:
                conf = to_scalar(box.conf) if hasattr(box, "conf") else 0.0
            except Exception:
                try:
                    conf = to_scalar(box.conf[0])
                except Exception:
                    conf = 0.0

            try:
                cls = int(to_scalar(box.cls) if hasattr(box, "cls") else to_scalar(box.label if hasattr(box, "label") else 0))
            except Exception:
                cls = 0

            label = names[cls] if (isinstance(names, list) and 0 <= cls < len(names)) else str(cls)

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [round(x, 2) for x in xyxy]
            })

            draw_box(annotated, xyxy, label, conf)

        return annotated, detections


# ---------------- UI Components ----------------

class ScrollableImageFrame(ctk.CTkFrame):
    """
    Scrollable grid of thumbnails with 4 columns per row.
    Clicking opens a larger centered viewer window (~80% of screen).
    Touchpad / mouse-wheel scrolling supported.
    """
    def __init__(self, master, width=800, height=500, thumb_size=(300, 180), cols=4, **kwargs):
        super().__init__(master, **kwargs)
        self._tk = tk
        self.thumb_size = thumb_size
        self.cols = cols

        # Canvas + vertical scrollbar
        self.canvas = self._tk.Canvas(self, width=width, height=height, bg="#111111", highlightthickness=0)
        self.v_scroll = self._tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        # inner frame for grid
        self.inner_frame = ctk.CTkFrame(self.canvas, fg_color=self.cget("fg_color"))
        self.inner_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Bind enter/leave so wheel events are active only when pointer is over widget
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        self._images: List[ImageTk.PhotoImage] = []
        self._full_image_refs = {}  # map widget -> full PIL image
        self._count = 0  # number of thumbnails added

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # make inner window same width as canvas
        self.canvas.itemconfig(self.inner_window, width=event.width)

    def clear(self):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self._images.clear()
        self._full_image_refs.clear()
        self._count = 0

    def add_card(self, pil_img: Image.Image):
        """
        Adds a thumbnail and arranges it in a grid with self.cols columns.
        """
        img = pil_img.copy()
        img.thumbnail(self.thumb_size)
        photo = ImageTk.PhotoImage(img)
        self._images.append(photo)  # keep reference

        btn = ctk.CTkButton(self.inner_frame, image=photo, text="",
                            width=self.thumb_size[0], height=self.thumb_size[1],
                            fg_color="#1a1a1a", corner_radius=8)
        # compute grid position
        row = self._count // self.cols
        col = self._count % self.cols

        # place in grid with padding; center each cell by adding sticky
        btn.grid(row=row, column=col, padx=10, pady=8)
        self._full_image_refs[btn] = pil_img.copy()
        self._count += 1

        # ensure columns expand evenly (so thumbnails align centered)
        for c in range(self.cols):
            self.inner_frame.grid_columnconfigure(c, weight=1, uniform="col")

        # Clicking opens viewer
        def on_click(b=btn):
            full = self._full_image_refs.get(b)
            if full is not None:
                self._open_full_image_centered(full)

        btn.configure(command=on_click)

    def _open_full_image_centered(self, pil_img: Image.Image):
        """Open a centered, large viewer window (~80% of screen size)."""
        # get screen size
        screen_w = self.winfo_toplevel().winfo_screenwidth()
        screen_h = self.winfo_toplevel().winfo_screenheight()

        win_w = int(screen_w * 0.8)
        win_h = int(screen_h * 0.8)

        # maintain aspect ratio of pil_img within win_w x win_h
        img_w, img_h = pil_img.width, pil_img.height
        scale = min(win_w / img_w, win_h / img_h, 1.0)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)

        win = self._tk.Toplevel(self)
        win.title("Image Viewer")
        win.geometry(f"{win_w}x{win_h}+{(screen_w - win_w)//2}+{(screen_h - win_h)//2}")  # center
        win.configure(bg="#000000")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)

        # canvas to display scaled image and allow scrolling if needed (but scaled to fit)
        c = self._tk.Canvas(win, bg="#000000", highlightthickness=0)
        c.grid(row=0, column=0, sticky="nsew")

        # Create a resized copy for display (to fit window while preserving original)
        display_img = pil_img.copy()
        if (disp_w, disp_h) != (img_w, img_h):
            display_img = display_img.resize((disp_w, disp_h), Image.LANCZOS)

        photo = ImageTk.PhotoImage(display_img)
        c.image_ref = photo
        # center image in canvas
        c.create_image(win_w // 2, win_h // 2, anchor="center", image=photo)
        c.config(scrollregion=(0, 0, win_w, win_h))

        # Optional: allow zoom with mouse wheel + Ctrl (simple implementation)
        def on_wheel(ev):
            # if Ctrl pressed, zoom. Note: platform differences exist; checking state for Control
            if (ev.state & 0x4) != 0:  # Control key mask (works on many platforms)
                # rebuild scaled image centered at cursor — keep simple: ignore for now
                pass
            else:
                # scroll vertically
                delta = -1 * (ev.delta // 120) if hasattr(ev, "delta") else (1 if ev.num == 5 else -1)
                c.yview_scroll(delta, "units")

        # bind mouse wheel for scrolling inside viewer
        c.bind_all("<MouseWheel>", on_wheel)
        c.bind_all("<Button-4>", lambda e: on_wheel(e))
        c.bind_all("<Button-5>", lambda e: on_wheel(e))

        # close bindings on window destroy
        def on_close():
            try:
                c.unbind_all("<MouseWheel>")
                c.unbind_all("<Button-4>")
                c.unbind_all("<Button-5>")
            except Exception:
                pass
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    # --- mouse wheel / touchpad support ---
    def _bind_mousewheel(self, event=None):
        # Windows and macOS use <MouseWheel> with event.delta
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        # Linux wheel events
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, event=None):
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except Exception:
            pass

    def _on_mousewheel_windows(self, event):
        # handle small deltas from touchpad too
        lines = int(-1 * (event.delta / 120)) if event.delta != 0 else ( -1 if event.delta > 0 else 1)
        if lines == 0:
            lines = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(lines, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


# ---------------- Main App ----------------

class App(ctk.CTk):
    def __init__(self, model_path, default_out="./output", device=None, imgsz=640):
        super().__init__()

        self.title("Hand Detection – Gloved / Bare")
        self.geometry("1100x750")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.detector = Detector(model_path, device=device, imgsz=imgsz)
        self.out_dir = Path(default_out)
        self.input_folder = None

        # --- Top UI ---
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=12, pady=12)

        # center_frame will hold the centered buttons
        center_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        center_frame.pack(anchor="center")

        self.btn_select = ctk.CTkButton(
            center_frame, text="Select Folder",
            command=self.select_folder, width=140, height=38
        )
        self.btn_select.grid(row=0, column=0, padx=(0, 12))

        self.btn_start = ctk.CTkButton(
            center_frame, text="Start Detection",
            command=self.start_detection, width=140, height=38
        )
        self.btn_start.grid(row=0, column=1, padx=(12, 0))

        # folder label centered below buttons
        self.lbl_folder = ctk.CTkLabel(top_frame, text="No folder selected")
        self.lbl_folder.pack(anchor="center", pady=(8, 0))

        self.progress = ctk.CTkProgressBar(top_frame)
        self.progress.pack(fill="x", padx=160, pady=(10, 0))  # margin left/right to align with centered buttons
        self.progress.set(0)

        self.lbl_status = ctk.CTkLabel(top_frame, text="Ready")
        self.lbl_status.pack(anchor="center", pady=(8, 0))

        # --- Scroll Area (thumbnails grid) ---
        self.scroll_frame = ScrollableImageFrame(self, thumb_size=(260, 160), cols=4)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- UI Actions ----------------

    def select_folder(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory()

        if folder:
            self.input_folder = Path(folder)
            self.after(0, lambda: self._on_folder_selected(folder))

    def _on_folder_selected(self, folder):
        self.lbl_folder.configure(text=str(folder))
        self.scroll_frame.clear()
        self.lbl_status.configure(text="Folder selected")

    def start_detection(self):
        if not self.input_folder:
            self.lbl_status.configure(text="Please select a folder first!")
            return

        self.btn_start.configure(state="disabled")
        thread = threading.Thread(target=self.run_detection, daemon=True)
        thread.start()

    def run_detection(self):
        try:
            imgs = [p for p in self.input_folder.rglob("*")
                    if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]

            if not imgs:
                self._thread_safe_status("No images found.")
                self._thread_safe_enable_start()
                return

            ensure_dir(self.out_dir)
            ensure_dir(self.out_dir / "logs")

            total = len(imgs)
            self._thread_safe_progress(0.0)
            self._thread_safe_status("Loading model...")

            try:
                self.detector.load()
            except Exception as e:
                self._thread_safe_status(f"Model load failed: {e}")
                self._thread_safe_enable_start()
                return

            for i, img_path in enumerate(imgs, start=1):
                try:
                    bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if bgr is None:
                        self._thread_safe_status(f"Warning: could not read {img_path.name}")
                        continue

                    if bgr.ndim == 3 and bgr.shape[2] == 4:
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

                    annotated, detections = self.detector.predict(bgr)

                    save_path = self.out_dir / f"{img_path.stem}_annotated.jpg"
                    cv2.imwrite(str(save_path), annotated)

                    with open(self.out_dir / "logs" / f"{img_path.stem}.json", "w") as f:
                        json.dump({"filename": img_path.name,
                                   "detections": detections}, f, indent=2)

                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)

                    # add only image thumbnail (no text)
                    self.after(0, lambda pil=pil: self.scroll_frame.add_card(pil))

                except Exception:
                    tb = traceback.format_exc()
                    print(f"Error processing {img_path}: {tb}")

                self._thread_safe_progress(i / total)
                self._thread_safe_status(f"Processing {i}/{total}")

            self._thread_safe_status("Done!")
        finally:
            self._thread_safe_enable_start()

    # --- helper methods to safely update UI from worker thread ---

    def _thread_safe_status(self, text: str):
        self.after(0, lambda: self.lbl_status.configure(text=text))

    def _thread_safe_progress(self, value: float):
        self.after(0, lambda: self.progress.set(value))

    def _thread_safe_enable_start(self):
        self.after(0, lambda: self.btn_start.configure(state="normal"))


# ------------------- Entry -------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="./best.pt", help="Path to YOLO model (.pt / .safetensors / folder)")
    p.add_argument("--out_dir", default="./output", help="Output directory")
    p.add_argument("--device", default=None, help="Device for inference, e.g. 'cpu', 'cuda:0'")
    p.add_argument("--imgsz", default=640, type=int, help="Inference image size")
    return p.parse_args()


def main():
    args = parse_args()
    app = App(args.model, args.out_dir, device=args.device, imgsz=args.imgsz)
    app.mainloop()


if __name__ == "__main__":
    main()
