
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from pca_py import process_image

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PCA Image Transformer")
        self.geometry("820x500")

        self.input_path = None
        self.output_path = None
        self.k = tk.IntVar(value=3)

        self.create_widgets()

    def create_widgets(self):
        control_frame = tk.Frame(self)
        control_frame.pack(pady=10)

        btn_open = tk.Button(control_frame, text="Open Image", command=self.open_image)
        btn_open.grid(row=0, column=0, padx=5)

        lbl_k = tk.Label(control_frame, text="Components:")
        lbl_k.grid(row=0, column=1)
        slider = tk.Scale(control_frame, from_=1, to=3, orient=tk.HORIZONTAL, variable=self.k)
        slider.grid(row=0, column=2)

        btn_run = tk.Button(control_frame, text="Run PCA", command=self.run_pca)
        btn_run.grid(row=0, column=3, padx=5)

        btn_save = tk.Button(control_frame, text="Save As", command=self.save_image)
        btn_save.grid(row=0, column=4, padx=5)

        # Image display canvases
        self.canvas_orig = tk.Canvas(self, width=350, height=350, bg='lightgray')
        self.canvas_orig.place(x=20, y=60)
        self.canvas_proc = tk.Canvas(self, width=350, height=350, bg='lightgray')
        self.canvas_proc.place(x=450, y=60)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.bmp *.jpg *.jpeg")])
        if path:
            self.input_path = path
            self.output_path = None
            self.display_image(path, self.canvas_orig, 'orig')

    def display_image(self, path, canvas, tag):
        img = Image.open(path)
        img.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(175, 175, image=photo)
        setattr(self, f'photo_{tag}', photo)

    def run_pca(self):
        if not self.input_path:
            messagebox.showwarning("No image", "Please open an image first.")
            return
        # generate default output path if not set
        if not self.output_path:
            base, ext = os.path.splitext(self.input_path)
            self.output_path = base + f"_pca{self.k.get()}" + ext
        try:
            process_image(self.input_path, self.output_path, self.k.get())
            self.display_image(self.output_path, self.canvas_proc, 'proc')
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_image(self):
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showwarning("No output", "Please run PCA first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=os.path.splitext(self.output_path)[1],
                                            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
        if path:
            os.replace(self.output_path, path)
            self.output_path = path
            messagebox.showinfo("Saved", f"Image saved to {path}")

if __name__ == "__main__":
    app = App()
    app.mainloop()