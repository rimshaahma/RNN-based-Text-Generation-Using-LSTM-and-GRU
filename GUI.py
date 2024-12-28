import tkinter as tk

# Create the main window
window = tk.Tk()
window.title("Text Generation with GRU and LSTM")

# Create a label and entry for the user input
label = tk.Label(window, text="Enter the starting text:")
label.pack()

start_string_entry = tk.Entry(window, width=50)
start_string_entry.pack()

# Function to handle text generation on button click
def generate_text_button_clicked():
    start_string = start_string_entry.get()
    generated_text = generate_text(gru_model, start_string, char_to_int, int_to_char, seq_length=100, num_generate=500)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, generated_text)

# Create a button to generate text
generate_button = tk.Button(window, text="Generate Text", command=generate_text_button_clicked)
generate_button.pack()

# Create a Text widget to display the generated text
output_text = tk.Text(window, height=10, width=50)
output_text.pack()

# Start the Tkinter event loop
window.mainloop()
