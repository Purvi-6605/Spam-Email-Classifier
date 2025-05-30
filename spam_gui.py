import tkinter as tk
from tkinter import messagebox, filedialog
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Predict and update label
def classify_text(text):
    if not text:
        messagebox.showwarning("Input error", "Please enter or attach email content.")
        return

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]

    if prediction == 1:
        result_label.config(text="🛑 SPAM", fg="red")
    else:
        result_label.config(text="✅ NOT SPAM", fg="green")

# Button handlers
def check_spam():
    text = text_input.get("1.0", tk.END).strip()
    classify_text(text)

def attach_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            text_input.delete("1.0", tk.END)
            text_input.insert(tk.END, content)
        classify_text(content)

def recheck():
    text = text_input.get("1.0", tk.END).strip()
    classify_text(text)

# GUI setup
root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("550x480")

tk.Label(root, text="Enter or attach email content:", font=("Arial", 14)).pack(pady=10)
text_input = tk.Text(root, height=12, width=65, wrap="word", font=("Arial", 11))
text_input.pack(padx=10, pady=10)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Check Spam", font=("Arial", 12), command=check_spam, bg="green", fg="white").pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Attach File", font=("Arial", 12), command=attach_file, bg="blue", fg="white").pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Recheck", font=("Arial", 12), command=recheck, bg="orange", fg="white").pack(side=tk.LEFT, padx=10)

# Output label
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
