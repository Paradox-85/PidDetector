import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

import PDFProcessor


class ModernPDFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Data Extractor")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.configure(bg="#f8f9fa")

        # Application state
        self.tree = None
        self.current_df = None
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")

        # Configure modern styling
        self.configure_styles()

        # Create main layout
        self.create_layout()

        # Setup drag and drop (assuming tkinterdnd2 is available)
        self.setup_drag_drop()

    def configure_styles(self):
        """Configure modern styling for ttk widgets"""
        self.style = ttk.Style()

        # Use a modern theme as base
        available_themes = self.style.theme_names()
        if 'vista' in available_themes:
            self.style.theme_use('vista')
        elif 'clam' in available_themes:
            self.style.theme_use('clam')

        # Configure custom styles
        self.style.configure("Modern.TFrame",
                             background="#ffffff",
                             relief="flat",
                             borderwidth=1)

        self.style.configure("Card.TFrame",
                             background="#ffffff",
                             relief="solid",
                             borderwidth=1)

        self.style.configure("Modern.TButton",
                             background="#ffffff",
                             foreground="#009DF0",
                             borderwidth=0,
                             focuscolor="none",
                             padding=(20, 10),
                             font=('Segoe UI', 10))

        self.style.map("Modern.TButton",
                       background=[('active', '#0088d1'),
                                   ('pressed', '#0071b3'),
                                   ('disabled', '#cccccc')])

        self.style.configure("Secondary.TButton",
                             background="#ffffff",
                             foreground="#009DF0",
                             borderwidth=0,
                             focuscolor="none",
                             padding=(15, 8),
                             font=('Segoe UI', 9))

        self.style.map("Secondary.TButton",
                       background=[('active', '#545b62'),
                                   ('pressed', '#3a3f44')])

        self.style.configure("Modern.Treeview",
                             background="#ffffff",
                             foreground="#212529",
                             rowheight=30,
                             fieldbackground="#ffffff",
                             font=('Segoe UI', 9),
                             borderwidth=1,
                             relief="solid")

        self.style.configure("Modern.Treeview.Heading",
                             background="#e9ecef",
                             foreground="#495057",
                             font=('Segoe UI', 9, 'bold'),
                             relief="flat",
                             borderwidth=1)

        self.style.map('Modern.Treeview',
                       background=[('selected', '#009DF0')],
                       foreground=[('selected', '#ffffff')])

        self.style.configure("Modern.Horizontal.TProgressbar",
                             background="#009DF0",
                             troughcolor="#e9ecef",
                             borderwidth=0,
                             lightcolor="#009DF0",
                             darkcolor="#009DF0")

    def create_layout(self):
        """Create the main application layout"""
        # Main container
        self.main_frame = ttk.Frame(self.root, style="Modern.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        self.create_header()

        # Content area
        self.content_frame = ttk.Frame(self.main_frame, style="Modern.TFrame")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))

        # Initially show upload UI
        self.show_upload_ui()

        # Footer with status bar
        self.create_footer()

    def create_header(self):
        """Create the application header"""
        header_frame = ttk.Frame(self.main_frame, style="Modern.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title and subtitle
        title_label = tk.Label(header_frame,
                               text="PDF Data Extractor",
                               font=('Segoe UI', 24, 'bold'),
                               bg="#f8f9fa",
                               fg="#212529")
        title_label.pack(anchor=tk.W)

        subtitle_label = tk.Label(header_frame,
                                  text="Extract tabular data from PDF documents",
                                  font=('Segoe UI', 11),
                                  bg="#f8f9fa",
                                  fg="#6c757d")
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))

        # Separator
        separator = ttk.Separator(header_frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=(15, 0))

    def create_footer(self):
        """Create the footer with status bar"""
        self.footer_frame = ttk.Frame(self.main_frame, style="Modern.TFrame")
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))

        # Status bar
        status_frame = ttk.Frame(self.footer_frame, style="Card.TFrame")
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(status_frame,
                                     textvariable=self.status_var,
                                     font=('Segoe UI', 9),
                                     bg="#ffffff",
                                     fg="#495057",
                                     anchor=tk.W,
                                     padx=10,
                                     pady=5)
        self.status_label.pack(fill=tk.X)

    def show_upload_ui(self):
        """Show the upload/drag-drop interface"""
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Upload card
        self.upload_card = ttk.Frame(self.content_frame, style="Card.TFrame")
        self.upload_card.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Upload content
        upload_content = ttk.Frame(self.upload_card, style="Modern.TFrame")
        upload_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Upload icon (using text as placeholder)
        icon_label = tk.Label(upload_content,
                              text="📄",
                              font=('Segoe UI', 48),
                              bg="#ffffff",
                              fg="#009DF0")
        icon_label.pack(pady=(0, 20))

        # Upload text
        upload_label = tk.Label(upload_content,
                                text="Drag and drop a PDF file here",
                                font=('Segoe UI', 16),
                                bg="#ffffff",
                                fg="#495057")
        upload_label.pack()

        # Or text
        or_label = tk.Label(upload_content,
                            text="or",
                            font=('Segoe UI', 12),
                            bg="#ffffff",
                            fg="#6c757d")
        or_label.pack(pady=(10, 15))

        # Browse button
        self.browse_button = ttk.Button(upload_content,
                                        text="Browse Files",
                                        command=self.browse_file,
                                        style="Modern.TButton")
        self.browse_button.pack()

        # Supported formats
        formats_label = tk.Label(upload_content,
                                 text="Supported format: PDF",
                                 font=('Segoe UI', 9),
                                 bg="#ffffff",
                                 fg="#6c757d")
        formats_label.pack(pady=(15, 0))

    def show_data_view(self, dataframe):
        """Show the data grid view"""
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Data view frame
        data_frame = ttk.Frame(self.content_frame, style="Modern.TFrame")
        data_frame.pack(fill=tk.BOTH, expand=True)

        # Toolbar
        self.create_toolbar(data_frame)

        # Data grid with scrollbars
        grid_frame = ttk.Frame(data_frame, style="Card.TFrame")
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Create treeview
        self.tree = ttk.Treeview(grid_frame, style="Modern.Treeview")

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(grid_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(grid_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        grid_frame.grid_rowconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(0, weight=1)

        # Configure treeview
        self.tree["columns"] = list(dataframe.columns)
        self.tree["show"] = "headings"

        for col in dataframe.columns:
            self.tree.heading(col, text=col, anchor=tk.W)
            self.tree.column(col, width=150, anchor=tk.W, minwidth=100)

        # Insert data
        for _, row in dataframe.iterrows():
            self.tree.insert("", "end", values=list(row))

        # Update status
        self.status_var.set(f"Loaded {len(dataframe)} rows × {len(dataframe.columns)} columns")

    def create_toolbar(self, parent):
        """Create toolbar with action buttons"""
        toolbar = ttk.Frame(parent, style="Modern.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 10))

        # Left side buttons
        left_frame = ttk.Frame(toolbar, style="Modern.TFrame")
        left_frame.pack(side=tk.LEFT)

        self.export_button = ttk.Button(left_frame,
                                        text="📊 Export to Excel",
                                        command=self.export_to_xlsx,
                                        style="Modern.TButton")
        self.export_button.pack(side=tk.LEFT, padx=(0, 10))

        # Right side buttons
        right_frame = ttk.Frame(toolbar, style="Modern.TFrame")
        right_frame.pack(side=tk.RIGHT)

        self.reset_button = ttk.Button(right_frame,
                                       text="🔄 Load New PDF",
                                       command=self.reset_ui,
                                       style="Secondary.TButton")
        self.reset_button.pack(side=tk.RIGHT)

    def show_progress_ui(self):
        """Show progress indicator"""
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Progress card
        progress_card = ttk.Frame(self.content_frame, style="Card.TFrame")
        progress_card.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Progress content
        progress_content = ttk.Frame(progress_card, style="Modern.TFrame")
        progress_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Progress icon
        icon_label = tk.Label(progress_content,
                              text="⚙️",
                              font=('Segoe UI', 48),
                              bg="#ffffff",
                              fg="#009DF0")
        icon_label.pack(pady=(0, 20))

        # Progress text
        progress_label = tk.Label(progress_content,
                                  text="Processing PDF...",
                                  font=('Segoe UI', 16),
                                  bg="#ffffff",
                                  fg="#495057")
        progress_label.pack()

        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_content,
                                            variable=self.progress_var,
                                            maximum=100,
                                            length=400,
                                            style="Modern.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=20)

        # Progress percentage
        self.progress_text = tk.Label(progress_content,
                                      text="0%",
                                      font=('Segoe UI', 12),
                                      bg="#ffffff",
                                      fg="#6c757d")
        self.progress_text.pack()

    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        try:
            # This would require tkinterdnd2 package
            # self.upload_card.drop_target_register(DND_FILES)
            # self.upload_card.dnd_bind('<<Drop>>', self.on_drop)
            pass
        except:
            # Fallback if drag-drop library not available
            pass

    def on_drop(self, event):
        """Handle dropped files"""
        if hasattr(event, 'data'):
            file_path = event.data.strip('{}')
            self.handle_file(file_path)

    def browse_file(self):
        """Open file browser dialog"""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=str(Path.home())
        )
        if file_path:
            self.handle_file(file_path)

    def handle_file(self, file_path):
        """Handle selected file"""
        if not file_path.lower().endswith('.pdf'):
            messagebox.showerror("Invalid File", "Please select a PDF file.")
            return

        self.status_var.set(f"Loading: {Path(file_path).name}")
        self.show_progress_ui()

        # Start processing in background thread
        threading.Thread(target=self.process_pdf, args=(file_path,), daemon=True).start()

    def process_pdf(self, file_path):
        """Process PDF file (mock implementation)"""
        try:
            # Mock progress updates
            for i in range(0, 101, 10):
                self.root.after(0, lambda p=i: self.update_progress(p))
                threading.Event().wait(0.2)  # Simulate processing time

            # Mock dataframe creation (replace with actual PDFProcessor call)
            import pandas as pd

            df = PDFProcessor.get_data_from_pdf_easyocr(file_path,self.update_progress)

            self.current_df = df
            self.root.after(0, lambda: self.show_data_view(df))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process PDF:\n{str(e)}"))
            self.root.after(0, self.show_upload_ui)

    def update_progress(self, percent):
        """Update progress bar"""
        self.progress_var.set(percent)
        if hasattr(self, 'progress_text'):
            self.progress_text.config(text=f"{percent}%")

    def export_to_xlsx(self):
        """Export data to Excel"""
        if self.current_df is None:
            messagebox.showerror("No Data", "No data available to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
            initialdir=str(Path.home())
        )

        if file_path:
            try:
                self.current_df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported successfully!\n{file_path}")
                self.status_var.set(f"Exported to: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def reset_ui(self):
        """Reset UI to initial state"""
        self.current_df = None
        self.progress_var.set(0)
        self.status_var.set("Ready")
        self.show_upload_ui()


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernPDFApp(root)
    root.mainloop()
