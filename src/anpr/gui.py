"""
gui.py
~~~~~~
Tkinter GUI wrapper for ANP-Recognition CLI workflows.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


_REPO_ROOT = Path(__file__).resolve().parents[2]


class ANPRGui(tk.Tk):
    """Desktop GUI for running ANP-Recognition image/camera flows."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        log_level: str = "INFO",
        save_dir: str | None = None,
        no_window: bool = False,
    ) -> None:
        super().__init__()
        self.title("ANP-Recognition GUI")
        self.geometry("880x640")
        self.minsize(760, 520)

        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._output_queue: queue.Queue[str] = queue.Queue()

        self.mode = tk.StringVar(value="image")
        self.config_path = tk.StringVar(value=config_path)
        self.log_level = tk.StringVar(value=log_level)
        self.save_dir = tk.StringVar(value=save_dir or "")
        self.no_window = tk.BooleanVar(value=no_window)
        self.image_input = tk.StringVar(value="")
        self.camera_source = tk.StringVar(value="0")
        self.max_frames = tk.StringVar(value="")

        self._build_ui()
        self._refresh_mode_fields()
        self.after(120, self._poll_output)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        options = ttk.LabelFrame(root, text="Options", padding=10)
        options.pack(fill="x")

        self._add_file_row(
            parent=options,
            row=0,
            label="Config",
            variable=self.config_path,
            title="Select config file",
            is_dir=False,
        )

        ttk.Label(options, text="Log level").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(
            options,
            textvariable=self.log_level,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", pady=6)

        self._add_file_row(
            parent=options,
            row=2,
            label="Save dir",
            variable=self.save_dir,
            title="Select output directory",
            is_dir=True,
            allow_empty=True,
        )

        ttk.Checkbutton(
            options,
            text="No OpenCV window (headless mode)",
            variable=self.no_window,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=6)

        mode = ttk.LabelFrame(root, text="Mode", padding=10)
        mode.pack(fill="x", pady=(10, 0))

        ttk.Radiobutton(
            mode,
            text="Image",
            variable=self.mode,
            value="image",
            command=self._refresh_mode_fields,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode,
            text="Camera/Video/Stream",
            variable=self.mode,
            value="camera",
            command=self._refresh_mode_fields,
        ).grid(row=0, column=1, sticky="w", padx=(18, 0))

        self.image_row_parent = mode
        self._add_file_row(
            parent=mode,
            row=1,
            label="Image path",
            variable=self.image_input,
            title="Select image file or folder",
            is_dir=False,
        )
        ttk.Button(
            mode,
            text="Browse folder",
            command=self._browse_image_folder,
            width=14,
        ).grid(row=1, column=3, padx=(6, 0), sticky="w")

        self.cam_source_label = ttk.Label(mode, text="Source")
        self.cam_source_label.grid(row=2, column=0, sticky="w", pady=6)
        self.cam_source_entry = ttk.Entry(mode, textvariable=self.camera_source, width=52)
        self.cam_source_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=6)
        ttk.Button(
            mode,
            text="Browse video",
            command=self._browse_video_file,
            width=14,
        ).grid(row=2, column=3, padx=(6, 0), sticky="w")

        self.max_frames_label = ttk.Label(mode, text="Max frames")
        self.max_frames_label.grid(row=3, column=0, sticky="w", pady=6)
        self.max_frames_entry = ttk.Entry(mode, textvariable=self.max_frames, width=18)
        self.max_frames_entry.grid(row=3, column=1, sticky="w", pady=6)

        mode.columnconfigure(1, weight=1)
        options.columnconfigure(1, weight=1)

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", pady=(12, 8))
        self.run_button = ttk.Button(buttons, text="Run", command=self._run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Clear output", command=self._clear_output).pack(side="left", padx=(8, 0))

        output_frame = ttk.LabelFrame(root, text="Console output", padding=10)
        output_frame.pack(fill="both", expand=True)
        self.output = ScrolledText(output_frame, wrap="word", height=18)
        self.output.pack(fill="both", expand=True)
        self.output.configure(state="disabled")

    def _add_file_row(
        self,
        parent: ttk.LabelFrame | ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        title: str,
        is_dir: bool,
        allow_empty: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=variable, width=52).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=6
        )

        def _browse() -> None:
            path = (
                filedialog.askdirectory(title=title)
                if is_dir
                else filedialog.askopenfilename(title=title)
            )
            if path:
                variable.set(path)
            elif not path and allow_empty:
                variable.set("")

        ttk.Button(parent, text="Browse", command=_browse, width=14).grid(
            row=row, column=3, padx=(6, 0), sticky="w"
        )

    def _browse_image_folder(self) -> None:
        path = filedialog.askdirectory(title="Select image directory")
        if path:
            self.image_input.set(path)

    def _browse_video_file(self) -> None:
        path = filedialog.askopenfilename(title="Select video file")
        if path:
            self.camera_source.set(path)

    def _refresh_mode_fields(self) -> None:
        image_mode = self.mode.get() == "image"
        image_state = "normal" if image_mode else "disabled"
        camera_state = "disabled" if image_mode else "normal"

        for child in self.image_row_parent.grid_slaves(row=1):
            try:
                child.configure(state=image_state)
            except tk.TclError:
                pass

        self.cam_source_label.configure(state=camera_state)
        self.cam_source_entry.configure(state=camera_state)
        self.max_frames_label.configure(state=camera_state)
        self.max_frames_entry.configure(state=camera_state)
        for child in self.image_row_parent.grid_slaves(row=2):
            if isinstance(child, ttk.Button):
                child.configure(state=camera_state)

    def _build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "main.py"),
            "--config",
            self.config_path.get().strip(),
            "--log-level",
            self.log_level.get().strip().upper() or "INFO",
        ]

        save_dir = self.save_dir.get().strip()
        if save_dir:
            cmd.extend(["--save-dir", save_dir])
        if self.no_window.get():
            cmd.append("--no-window")

        mode = self.mode.get()
        cmd.append(mode)
        if mode == "image":
            image_path = self.image_input.get().strip()
            if not image_path:
                raise ValueError("Image path is required for image mode.")
            cmd.append(image_path)
        else:
            source = self.camera_source.get().strip() or "0"
            cmd.extend(["--source", source])
            max_frames = self.max_frames.get().strip()
            if max_frames:
                if not max_frames.isdigit() or int(max_frames) <= 0:
                    raise ValueError("Max frames must be a positive integer.")
                cmd.extend(["--max-frames", max_frames])

        return cmd

    def _run(self) -> None:
        if self._process and self._process.poll() is None:
            messagebox.showinfo("Running", "A process is already running.")
            return

        try:
            cmd = self._build_command()
        except ValueError as exc:
            messagebox.showerror("Invalid options", str(exc))
            return

        self._append_output("$ " + " ".join(cmd) + "\n")

        self._process = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        if not self._process or not self._process.stdout:
            return

        for line in self._process.stdout:
            self._output_queue.put(line)
        exit_code = self._process.wait()
        self._output_queue.put(f"\n[Process exited with code {exit_code}]\n")
        self._output_queue.put("__PROCESS_DONE__")

    def _poll_output(self) -> None:
        while True:
            try:
                item = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__PROCESS_DONE__":
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self._process = None
            else:
                self._append_output(item)
        self.after(120, self._poll_output)

    def _append_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _clear_output(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    def _stop(self) -> None:
        if not self._process or self._process.poll() is not None:
            return
        self._process.terminate()
        self._append_output("\n[Stopping process...]\n")

    def _on_close(self) -> None:
        self._stop()
        self.destroy()


def launch_gui(
    config_path: str = "config/config.yaml",
    log_level: str = "INFO",
    save_dir: str | None = None,
    no_window: bool = False,
) -> None:
    """Launch the ANPR Tkinter GUI."""
    app = ANPRGui(
        config_path=config_path,
        log_level=log_level,
        save_dir=save_dir,
        no_window=no_window,
    )
    app.mainloop()
