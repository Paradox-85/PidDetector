import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

import PDFProcessor


class PDFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Drag & Drop or Browse")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        self.tree = None
        self.export_button = None
        self.reset_button = None
        self.current_df = None
        self.progress_canvas = None

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Treeview",
                             background="#34495e",
                             foreground="white",
                             rowheight=25,
                             fieldbackground="#34495e",
                             font=('Verdana', 10))
        self.style.map('Treeview', background=[('selected', '#1abc9c')], foreground=[('selected', 'black')])
        self.style.configure('TButton',
                             background='#33ADEF',
                             foreground='#FFFFFF',
                             focusthickness=3,
                             borderwidth=0,
                             focuscolor='#33ADEF',
                             font=('Verdana', 13),
                             padding=(16, 8))
        self.style.map('TButton',
                       background=[('active', '#28a0dc'), ('pressed', '#1c7fc9')],
                       foreground=[('active', '#FFFFFF')])

        self.create_drag_drop_ui()

        self.path_label = tk.Label(root, text="", anchor="w", bg="#2c3e50", fg="white", font=("Helvetica", 10))
        self.path_label.place(x=10, y=10)
        self.path_label.lower()

    def create_drag_drop_ui(self):
        self.drop_frame = tk.Frame(self.root, bg="#34495e", width=480, height=360, relief=tk.RIDGE, bd=2)
        self.drop_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.label = tk.Label(self.drop_frame, text="Drag and drop a PDF here",
                              bg="#34495e", fg="white", font=("Helvetica", 16, 'bold'))
        self.label.pack(pady=(80, 10))

        self.browse_button = ttk.Button(self.drop_frame, text="Browse PDF", command=self.browse_file)
        self.browse_button.pack(pady=10)

        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)

    def destroy_drag_drop_ui(self):
        if self.drop_frame:
            self.drop_frame.destroy()

    def on_drop(self, event):
        file_path = event.data.strip("{}")
        self.handle_file(file_path)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.handle_file(file_path)

    def handle_file(self, file_path):
        if not file_path.lower().endswith(".pdf"):
            self.label.config(text="Error: Only PDF files are supported", fg="red")
            return

        self.show_progress_bar()

        threading.Thread(target=self.load_pdf, args=(file_path,), daemon=True).start()

    def load_pdf(self, file_path):
        try:
            df = PDFProcessor.get_data_from_pdf_easyocr(file_path, self.update_progress)
        except Exception as e:
            self.label.config(text=f"Error loading PDF: {e}", fg="red")
            return

        self.current_df = df

        self.root.after(0, self.progress_canvas.destroy)
        self.root.after(0, lambda: self.path_label.config(text=f"PDF Loaded: {file_path}", fg="white"))
        self.root.after(0, self.path_label.lift)
        self.root.after(0, self.destroy_drag_drop_ui)
        self.root.after(0, lambda: self.show_data_grid(df))
        self.root.after(0, self.show_export_button)
        self.root.after(0, self.show_reset_button)

    def show_progress_bar(self):
        self.progress_canvas = tk.Canvas(self.root, width=500, height=450, bg="#2c3e50", highlightthickness=0)
        self.progress_canvas.place(relx=0.5, rely=0.7, anchor="center")

        self.progress_bg = self.progress_canvas.create_rectangle(50, 70, 450, 90, fill="#e0e0e0", width=0)
        self.progress_fg = self.progress_canvas.create_rectangle(50, 70, 50, 90, fill="#1abc9c", width=0)
        self.progress_text = self.progress_canvas.create_text(250, 45, text="0%", fill="white", font=("Helvetica", 24, "bold"))
        self.progress_subtext = self.progress_canvas.create_text(250, 110, text="Please wait while the application is loading...",
                                                                 fill="#cccccc", font=("Helvetica", 10))

    def update_progress(self, percent):
        percent = min(max(percent, 0), 100)
        new_width = 50 + int(4 * percent)
        self.progress_canvas.coords(self.progress_fg, 50, 70, new_width, 90)
        self.progress_canvas.itemconfigure(self.progress_text, text=f"{percent}%")
        self.root.update_idletasks()

    def show_data_grid(self, dataframe):
        self.tree = ttk.Treeview(self.root)
        self.tree.place(x=10, y=40, width=780, height=480)

        self.tree["columns"] = list(dataframe.columns)
        self.tree["show"] = "headings"

        for col in dataframe.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')

        for _, row in dataframe.iterrows():
            self.tree.insert("", "end", values=list(row))

    def show_export_button(self):
        self.export_button = ttk.Button(self.root, text="Export to XLSX", command=self.export_to_xlsx)
        self.export_button.place(x=10, y=530)

    def show_reset_button(self):
        self.reset_button = ttk.Button(self.root, text="Reset", command=self.reset_ui)
        self.reset_button.place(x=200, y=530)

    def export_to_xlsx(self):
        if self.current_df is None:
            messagebox.showerror("Error", "No PDF data loaded to export.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save as"
        )
        if save_path:
            try:
                self.current_df.to_excel(save_path, index=False)
                messagebox.showinfo("Success", f"File exported successfully:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export file:\n{e}")

    def reset_ui(self):
        if self.tree:
            self.tree.destroy()
            self.tree = None
        if self.export_button:
            self.export_button.destroy()
            self.export_button = None
        if self.reset_button:
            self.reset_button.destroy()
            self.reset_button = None

        self.path_label.config(text="")
        self.path_label.lower()

        self.current_df = None

        self.create_drag_drop_ui()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = PDFApp(root)
    root.mainloop()