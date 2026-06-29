from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, replace
from tkinter import ttk
from typing import Optional


@dataclass
class AlbedoAppSettings:
    method: str = "luminance"  # luminance | per_channel | gaussian_ratio | careaga
    # classical
    sigma: float = 25.0
    smooth_shading_sigma: float = 14.0
    # Per-channel 2-98% stretch (garish). Default off = natural color relations.
    vivid: bool = False
    # Luminance Retinex: 8-bit preview only (log(A)=log I − log S is unchanged). Lower = darker.
    view_gain: float = 0.25
    max_display_scale: float = 48.0
    # Flatter, less neon preview (luminance: gray-world on the ratio; all classical: L*a*b* desat)
    gray_world: bool = True
    chroma_damp: float = 0.5
    # albedo | raw_albedo | shading_albedo | raw_shading_albedo
    view_layout: str = "shading_albedo"
    # careaga
    careaga_quality: str = "balanced"  # fast | balanced | best
    careaga_version: str = "v2"  # v2 | v2.1
    device: str = "auto"  # auto | cpu | cuda | mps
    max_side: int = 384
    careaga_stride: int = 2
    pipeline_stage: int = 3  # 3=albedo only, 4=+diffuse (slower)
    # camera
    width: int = 640
    height: int = 480
    cam: int = 0
    buffer_size: int = 1
    disable_auto_exposure: bool = False


def _apply_careaga_preset(settings: AlbedoAppSettings) -> AlbedoAppSettings:
    q = (settings.careaga_quality or "balanced").lower()
    if q == "fast":
        return replace(settings, max_side=256, careaga_stride=4)
    if q == "best":
        return replace(settings, max_side=512, careaga_stride=1)
    return replace(settings, max_side=384, careaga_stride=2)  # balanced


def _on_quality_change(
    v: str,
    max_side_var: tk.IntVar,
    stride_var: tk.IntVar,
) -> None:
    if v == "fast":
        max_side_var.set(256)
        stride_var.set(4)
    elif v == "best":
        max_side_var.set(512)
        stride_var.set(1)
    else:
        max_side_var.set(384)
        stride_var.set(2)


def show_albedo_settings(
    initial: Optional[AlbedoAppSettings] = None,
) -> Optional[AlbedoAppSettings]:
    """
    Show modal dialog. Returns :class:`AlbedoAppSettings` on OK, ``None`` on cancel.
    """
    start = initial or AlbedoAppSettings()
    if start.method == "careaga":
        start = _apply_careaga_preset(start)

    root = tk.Tk()
    root.title("Albedo camera — settings")
    root.resizable(True, True)
    root.minsize(480, 420)

    out: list[Optional[AlbedoAppSettings]] = [None]

    m_names = {
        "luminance": "Classical — luminance log-Retinex (recommended)",
        "per_channel": "Classical — Multi-Scale Retinex (MSR, σ=15,80,250)",
        "gaussian_ratio": "Classical — fast R/(blur+eps)",
        "careaga": "Learned — Careaga & Aksoy (Colorful, needs Intrinsic)",
    }
    name_to_m = {v: k for k, v in m_names.items()}

    f_main = ttk.Frame(root, padding=10)
    f_main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(f_main, text="Method", font=("", 11, "bold")).grid(row=0, column=0, sticky="w")
    var_method = tk.StringVar(value=m_names.get(start.method, m_names["luminance"]))
    cb_method = ttk.Combobox(
        f_main,
        textvariable=var_method,
        values=tuple(m_names.values()),
        state="readonly",
        width=50,
    )
    cb_method.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
    f_main.columnconfigure(1, weight=1)

    f_classic = ttk.LabelFrame(f_main, text="Classical options", padding=6)
    f_classic.grid(row=1, column=0, columnspan=3, sticky="ew", pady=8)
    ttk.Label(
        f_classic,
        text="σ (Gaussian R/(blur); MSR uses fixed 15, 80, 250)",
    ).grid(row=0, column=0, sticky="w")
    var_sigma = tk.DoubleVar(value=start.sigma)
    ttk.Spinbox(f_classic, from_=1, to=99, textvariable=var_sigma, width=8).grid(
        row=0, column=1, sticky="w", padx=4
    )
    ttk.Label(f_classic, text="Luminance: smooth log-shading σ (0=off)").grid(
        row=1, column=0, sticky="w"
    )
    var_smooth = tk.DoubleVar(value=start.smooth_shading_sigma)
    ttk.Spinbox(f_classic, from_=0, to=80, textvariable=var_smooth, width=8).grid(
        row=1, column=1, sticky="w", padx=4
    )
    var_vivid = tk.BooleanVar(value=start.vivid)
    ttk.Checkbutton(
        f_classic,
        text="Vivid (per-channel tone map, oversaturated / neon look)",
        variable=var_vivid,
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=4)

    ttk.Label(
        f_classic,
        text="Luminance: preview gain (0.2–0.6, not part of Retinex math)",
    ).grid(row=3, column=0, sticky="w", pady=2)
    var_vg = tk.DoubleVar(value=float(start.view_gain))
    ttk.Spinbox(
        f_classic, from_=0.1, to=0.8, increment=0.02, textvariable=var_vg, width=8
    ).grid(row=3, column=1, sticky="w", pady=2)
    ttk.Label(f_classic, text="(Eq. is log A = log I − log S; this only maps to screen)").grid(
        row=3, column=2, columnspan=2, sticky="w", padx=4
    )
    ttk.Label(f_classic, text="Chroma damp (0=gray, 1=raw preview)").grid(
        row=4, column=0, sticky="w", pady=2
    )
    var_cd = tk.DoubleVar(value=float(start.chroma_damp))
    ttk.Spinbox(
        f_classic, from_=0.0, to=1.0, increment=0.05, textvariable=var_cd, width=8
    ).grid(row=4, column=1, sticky="w", pady=2)
    var_gw = tk.BooleanVar(value=start.gray_world)
    ttk.Checkbutton(
        f_classic,
        text="Gray-world (luminance: balance channel means on ratio)",
        variable=var_gw,
    ).grid(row=4, column=2, columnspan=2, sticky="w", padx=4)

    f_care = ttk.LabelFrame(f_main, text="Learned (Careaga) — quality / speed", padding=6)
    f_care.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)

    ttk.Label(f_care, text="Preset (sets max input side + frame stride)").grid(
        row=0, column=0, sticky="w"
    )
    var_q = tk.StringVar(value=start.careaga_quality)
    var_max = tk.IntVar(value=start.max_side)
    var_stride = tk.IntVar(value=start.careaga_stride)

    def on_q(*_) -> None:
        _on_quality_change(var_q.get().lower(), var_max, var_stride)

    ttk.Radiobutton(
        f_care, text="Fast (256px, stride 4)", variable=var_q, value="fast", command=on_q
    ).grid(row=0, column=1, padx=4)
    ttk.Radiobutton(
        f_care, text="Balanced (384, 2)", variable=var_q, value="balanced", command=on_q
    ).grid(row=0, column=2, padx=4)
    ttk.Radiobutton(
        f_care, text="Best (512, 1)", variable=var_q, value="best", command=on_q
    ).grid(row=0, column=3, padx=4)

    ttk.Label(f_care, text="Max input side (px)").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Spinbox(f_care, from_=128, to=1920, textvariable=var_max, width=8).grid(
        row=1, column=1, sticky="w", pady=4
    )
    ttk.Label(f_care, text="Run network every N frames").grid(row=1, column=2, sticky="e", padx=8)
    ttk.Spinbox(f_care, from_=1, to=30, textvariable=var_stride, width=6).grid(
        row=1, column=3, sticky="w", pady=4
    )

    ttk.Label(f_care, text="Weight pack").grid(row=2, column=0, sticky="w")
    var_ver = tk.StringVar(value=start.careaga_version)
    ttk.Combobox(
        f_care, textvariable=var_ver, values=("v2", "v2.1"), state="readonly", width=6
    ).grid(row=2, column=1, sticky="w", pady=2)

    ttk.Label(f_care, text="Device").grid(row=2, column=2, sticky="e", padx=8)
    var_dev = tk.StringVar(value=start.device)
    ttk.Combobox(
        f_care,
        textvariable=var_dev,
        values=("auto", "cpu", "cuda", "mps"),
        state="readonly",
        width=8,
    ).grid(row=2, column=3, sticky="w", pady=2)

    ttk.Label(f_care, text="Pipeline").grid(row=3, column=0, sticky="w", pady=4)
    var_stage = tk.StringVar(value=str(int(start.pipeline_stage)))
    ttk.Combobox(
        f_care,
        textvariable=var_stage,
        values=("3", "4"),
        state="readonly",
        width=4,
    ).grid(row=3, column=1, sticky="w", pady=4)
    ttk.Label(
        f_care,
        text="3 = albedo only (faster). 4 = +diffuse+residual (slower).",
    ).grid(row=3, column=2, columnspan=2, sticky="w")

    f_cam = ttk.LabelFrame(f_main, text="Camera", padding=6)
    f_cam.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
    ttk.Label(f_cam, text="Width").grid(row=0, column=0, sticky="w")
    var_w = tk.IntVar(value=start.width)
    ttk.Spinbox(f_cam, from_=160, to=1920, textvariable=var_w, width=7).grid(
        row=0, column=1, padx=4
    )
    ttk.Label(f_cam, text="Height").grid(row=0, column=2, sticky="e", padx=6)
    var_h = tk.IntVar(value=start.height)
    ttk.Spinbox(f_cam, from_=120, to=1200, textvariable=var_h, width=7).grid(
        row=0, column=3, padx=4
    )
    ttk.Label(f_cam, text="Device index").grid(row=0, column=4, sticky="e", padx=6)
    var_cam = tk.IntVar(value=start.cam)
    ttk.Spinbox(f_cam, from_=0, to=8, textvariable=var_cam, width=4).grid(
        row=0, column=5, padx=4
    )

    var_buf = tk.IntVar(value=start.buffer_size)
    ttk.Label(f_cam, text="Buffer").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Spinbox(f_cam, from_=1, to=4, textvariable=var_buf, width=4).grid(
        row=1, column=1, sticky="w", pady=4
    )
    layout_choices: dict[str, str] = {
        "albedo": "Albedo only",
        "raw_albedo": "input | albedo",
        "shading_albedo": "Y(log I) shading | albedo (default)",
        "raw_shading_albedo": "input | shading | albedo",
    }
    ttk.Label(f_cam, text="Preview layout").grid(row=1, column=2, sticky="e", padx=4)
    var_layout = tk.StringVar(value=layout_choices.get(start.view_layout, layout_choices["shading_albedo"]))
    ttk.Combobox(
        f_cam,
        textvariable=var_layout,
        values=tuple(layout_choices.values()),
        state="readonly",
        width=32,
    ).grid(row=1, column=3, columnspan=3, sticky="w", padx=2, pady=4)
    var_noae = tk.BooleanVar(value=start.disable_auto_exposure)
    ttk.Checkbutton(
        f_cam, text="Disable auto-exposure (driver)", variable=var_noae
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

    f_btn = ttk.Frame(f_main)
    f_btn.grid(row=4, column=0, columnspan=3, pady=12)

    def toggle_frames(*_args) -> None:
        key = name_to_m.get(var_method.get(), "luminance")
        if key == "careaga":
            f_classic.grid_remove()
            f_care.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        else:
            f_care.grid_remove()
            f_classic.grid(row=1, column=0, columnspan=3, sticky="ew", pady=8)

    cb_method.bind("<<ComboboxSelected>>", toggle_frames)

    def on_ok() -> None:
        key = name_to_m.get(var_method.get(), "luminance")
        name_to_layout = {v: k for k, v in layout_choices.items()}
        vl = name_to_layout.get(var_layout.get(), "shading_albedo")
        out[0] = AlbedoAppSettings(
            method=key,
            sigma=float(var_sigma.get()),
            smooth_shading_sigma=float(var_smooth.get()),
            vivid=bool(var_vivid.get()),
            view_gain=float(var_vg.get()),
            max_display_scale=start.max_display_scale,
            gray_world=bool(var_gw.get()),
            chroma_damp=float(var_cd.get()),
            view_layout=vl,
            careaga_quality=var_q.get().lower(),
            careaga_version=var_ver.get().strip() or "v2",
            device=var_dev.get() or "auto",
            max_side=int(var_max.get()),
            careaga_stride=int(var_stride.get()),
            pipeline_stage=int(float(var_stage.get())),
            width=int(var_w.get()),
            height=int(var_h.get()),
            cam=int(var_cam.get()),
            buffer_size=int(var_buf.get()),
            disable_auto_exposure=bool(var_noae.get()),
        )
        if out[0].method == "careaga" and not (3 <= out[0].pipeline_stage <= 4):
            out[0] = replace(out[0], pipeline_stage=3)
        root.destroy()

    def on_cancel() -> None:
        out[0] = None
        root.destroy()

    ttk.Button(f_btn, text="Start camera", command=on_ok).pack(side=tk.LEFT, padx=6)
    ttk.Button(f_btn, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=6)

    ttk.Label(
        f_main,
        text="Faster learning path: lower max side, higher stride, pipeline stage 3 (albedo only).",
        font=("", 9),
    ).grid(row=5, column=0, columnspan=3, pady=4, sticky="w")

    # initial visibility
    toggle_frames()

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    return out[0]
