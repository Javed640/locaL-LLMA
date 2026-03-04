def select_model(self):
    use_ollama = messagebox.askyesno(
        "Select Model",
        "Do you want to use an Ollama model name (e.g., llama3:8b)?\n\n"
        "Yes = Ollama model name\nNo = Select a local .gguf file",
    )

    if use_ollama:
        name = simpledialog.askstring(
            "Ollama Model",
            "Enter Ollama model name (example: llama3:8b)\n"
            "Tip: run `ollama pull llama3:8b` first.",
        )
        if name:
            self.model_path = name.strip()
            messagebox.showinfo("Model Selected", f"Model set to (Ollama):\n{self.model_path}")
        return

    path = filedialog.askopenfilename(
        title="Select Model",
        initialdir="models",
        filetypes=[("GGUF Model", "*.gguf"), ("All files", "*.*")],
    )
    if path:
        self.model_path = path
        messagebox.showinfo("Model Selected", f"Model set to:\n{path}")