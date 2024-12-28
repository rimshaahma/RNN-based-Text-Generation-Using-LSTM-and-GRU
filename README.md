
# RNN-based Text Generation Using LSTM and GRU

This project demonstrates how to build and train Recurrent Neural Networks (RNNs) for text generation using both LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) layers. The project is based on training models using a Shakespeare dataset for generating text that mimics Shakespearean writing.

### **Table of Contents**
1. [Project Overview](#project-overview)
2. [Steps Involved](#steps-involved)
    1. [Step 1: Import Required Libraries](#step-1-import-required-libraries)
    2. [Step 2: Prepare the Data](#step-2-prepare-the-data)
    3. [Step 3: Build the RNN Model](#step-3-build-the-rnn-model)
    4. [Step 4: Train the Model](#step-4-train-the-model)
    5. [Step 5: Generate Text Using the Trained Model](#step-5-generate-text-using-the-trained-model)
    6. [Step 6: Experiment with GRU and Compare](#step-6-experiment-with-gru-and-compare)
    7. [Step 7: Compare LSTM and GRU](#step-7-compare-lstm-and-gru)
    8. [Step 8: (Optional) Build a GUI for Text Generation](#step-8-optional-build-a-gui-for-text-generation)
3. [Conclusion](#conclusion)
4. [Future Improvements](#future-improvements)
5. [Dependencies](#dependencies)

---

## **Project Overview**

This project focuses on building an RNN-based text generation model using both **LSTM** and **GRU** layers. The task is to generate coherent Shakespearean-style text based on a given starting phrase. The Shakespeare dataset, which consists of his works, is used to train the models.

We will:
1. Preprocess the dataset.
2. Build and train a model using LSTM and GRU.
3. Compare performance between LSTM and GRU models.
4. Use the trained models to generate new text.

---

## **Steps Involved**

### **Step 1: Import Required Libraries**

We begin by importing necessary libraries, which include TensorFlow for model building, NumPy for numerical operations, and Matplotlib for plotting the results.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

- **`tensorflow`**: Used for building and training the deep learning model.
- **`numpy`**: A library for numerical computing, particularly for handling arrays.
- **`matplotlib.pyplot`**: Used to visualize the training process.

### **Step 2: Prepare the Data**

In this step, we load the dataset and prepare it for training by encoding characters into integers and splitting the text into sequences for the RNN.

```python
# Load the dataset
text = open('shakespeare.txt', 'r').read()

# Prepare the character mappings
vocab = sorted(set(text))  # List of unique characters in the dataset
char_to_int = {char: i for i, char in enumerate(vocab)}  # Mapping characters to integers
int_to_char = {i: char for i, char in enumerate(vocab)}  # Reverse mapping

# Convert text to integer sequences
encoded_text = [char_to_int[char] for char in text]

# Define sequence length and batch size
seq_length = 100  # The length of the sequences
batch_size = 64
```

#### **Explanation**:
- **`text`**: Raw Shakespeare text file loaded into a string.
- **`vocab`**: The list of unique characters found in the text.
- **`char_to_int`** and **`int_to_char`**: These dictionaries map characters to integers and vice versa. They help convert the characters into numeric format for feeding into the neural network.
- **`encoded_text`**: A list of integers representing the input text, which is required for the neural network.
- **`seq_length`**: Length of the sequences that the model will learn from.

### **Step 3: Build the RNN Model**

Now, we build a Recurrent Neural Network (RNN) model. This model uses **LSTM** or **GRU** layers to process sequential data.

```python
def build_model(vocab_size, seq_length, batch_size):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 256, batch_input_shape=[batch_size, seq_length]),
        layers.LSTM(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

#### **Explanation**:
- **Embedding Layer**: Converts integer sequences into dense vectors of fixed size.
- **LSTM Layer**: The core of the model, capturing long-term dependencies in the text. We set `return_sequences=True` to output the sequence of hidden states for each time step.
- **Dense Layer**: A fully connected layer with a softmax activation that outputs a probability distribution for each character.

### **Step 4: Train the Model**

We split the dataset into training and validation sets, then use the `fit` method to train the model.

```python
# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))
```

#### **Explanation**:
- **`x_train` and `y_train`**: The input and target sequences used for training.
- **`epochs`**: Number of times the model will see the entire dataset.
- **`batch_size`**: The number of samples per gradient update.
- **`validation_data`**: A tuple of validation data to monitor the model’s performance on unseen data.

### **Step 5: Generate Text Using the Trained Model**

Once the model is trained, we can generate text by feeding a prompt (starting string) into the model and having it predict the next character iteratively.

```python
def generate_text(model, start_string, char_to_int, int_to_char, seq_length, num_generate=1000):
    # Convert start string to integer sequence
    input_eval = [char_to_int[char] for char in start_string]
    input_eval = np.expand_dims(input_eval, 0)  # Convert to batch dimension

    generated_text = start_string
    model.reset_states()

    for i in range(num_generate):
        predictions = model.predict(input_eval, verbose=0)
        predicted_id = tf.random.categorical(predictions[0], num_samples=1)[-1, 0].numpy()
        generated_text += int_to_char[predicted_id]
        input_eval = np.expand_dims([predicted_id], 0)
    
    return generated_text
```

#### **Explanation**:
- **`input_eval`**: A sequence of integers representing the start string.
- **`predictions`**: The predicted probabilities for the next character in the sequence.
- **`predicted_id`**: The index of the character with the highest probability.

### **Step 6: Experiment with GRU and Compare**

Now, we implement the GRU model, similar to the LSTM model, to see if it can provide better results in terms of training time or text generation quality.

```python
def build_gru_model(vocab_size, seq_length, batch_size):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 256, batch_input_shape=[batch_size, seq_length]),
        layers.GRU(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

#### **Explanation**:
- **GRU Layer**: Similar to LSTM, but with fewer parameters. It is often faster to train but still effective at capturing dependencies in sequential data.

### **Step 7: Compare LSTM and GRU**

Once both models are trained, you can compare their performance (in terms of loss and text generation) to evaluate which one performs better.

```python
# Generate text with LSTM
generated_text_lstm = generate_text(lstm_model, "Shall I compare thee", char_to_int, int_to_char, seq_length=100, num_generate=500)
print("Generated Text (LSTM):", generated_text_lstm)

# Generate text with GRU
generated_text_gru = generate_text(gru_model, "Shall I compare thee", char_to_int, int_to_char, seq_length=100, num_generate=500)
print("Generated Text (GRU):", generated_text_gru)
```

#### **Explanation**:
- **Text Comparison**: After generating text from both models, we can manually compare the results to see which model generates more coherent and meaningful text.

### **Step 8: (Optional) Build a GUI for Text Generation**

This step shows how to build a graphical user interface (GUI) for easier interaction with the text generation model. You can use **Tkinter** to create a simple window where you input a prompt and see the generated text.

```python
import tkinter as tk

# Create the main window
window = tk.Tk()
window.title("Text Generation with GRU and LSTM")

# Create a label and entry for the user input
label = tk.Label(window, text="Enter the starting text:")
label.pack()

start_string_entry = tk.Entry(window, width=50)
start_string_entry.pack()

# Function to

 handle text generation on button click
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
```

#### **Explanation**:
- **Tkinter**: This library helps create graphical applications. We use it here to create a simple interface for text generation.

---

## **Conclusion**

This project demonstrates how to build RNN-based models for text generation using LSTM and GRU layers. Both models were trained on a Shakespeare dataset, and we generated text based on user input. The project also compares the performance of LSTM and GRU and provides a GUI for easy interaction.

---

## **Future Improvements**

1. **Hyperparameter Tuning**: Experiment with different numbers of layers, units per layer, and activation functions to improve performance.
2. **Text Preprocessing**: Enhance text preprocessing with additional cleaning (e.g., removing unwanted characters or words).
3. **Model Evaluation**: Add more metrics to evaluate text generation quality (e.g., BLEU score, perplexity).
4. **Pre-trained Models**: Use pre-trained models for transfer learning to speed up training and improve performance.

---

## **Dependencies**

- **TensorFlow** (>=2.0): For building and training deep learning models.
- **NumPy**: For handling numerical data.
- **Matplotlib**: For plotting training loss and comparing results.
- **Tkinter**: For GUI (if you decide to implement the GUI part).

To install the required dependencies, run:

```bash
pip install tensorflow numpy matplotlib tk
```

