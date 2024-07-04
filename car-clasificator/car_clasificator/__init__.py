import tkinter as tk
from tkinter import filedialog
import concurrent.futures
from car_detector import video_process, image_process


# Function to handle file selection and storage
def select_files():
    filetypes = [
        ("MP4 files", "*.mp4"),
        ("JPEG files", "*.jpg"),
        ("All files", "*.*")
    ]
    files = filedialog.askopenfilenames(title="Select Files", filetypes=filetypes)
    if files:
        processing_label.config(text="Image is being processed")
        root.update_idletasks()  # Update the label immediately
        try:
            with open("stored_files.txt", "a") as f:
                for file in files:
                    f.write(file + "\n")
            selected_file_text.config(state=tk.NORMAL)  # Enable editing
            selected_file_text.delete(1.0, tk.END)  # Clear the text box
            selected_file_text.insert(tk.END, files[0])  # Insert first file name
            selected_file_text.config(state=tk.DISABLED)  # Make read-only again
        except Exception as e:
            processing_label.config(text=f"Error: {e}")
        finally:
            processing_label.config(text="")  # Clear the processing label


# Function to handle directory selection
def select_directory():
    directory = filedialog.askdirectory(title="Select Directory")
    if directory:
        selected_directory_text.config(state=tk.NORMAL)  # Enable editing
        selected_directory_text.delete(1.0, tk.END)  # Clear the text box
        selected_directory_text.insert(tk.END, directory)  # Insert selected directory
        selected_directory_text.config(state=tk.DISABLED)  # Make read-only again


# Function to process media files
def process_media():
    is_completed = False

    # Display processing message
    processing_label.config(text="MEDIA IS BEING PROCESSED")
    processing_label.pack(pady=20)  # Show the label under the process_button
    root.update_idletasks()  # Ensure label update is immediate

    is_video = False
    file_path = selected_file_text.get("1.0", "end-1c")
    if file_path.endswith(".mp4"):
        is_video = True
    result_path = selected_directory_text.get("1.0", "end-1c")
    name = file_name_text.get("1.0", "end-1c")
    full_path = result_path + "/" + name

    def process_task(in_path, out_path, is_video):
        if is_video:
            video_process(in_path, out_path)
            return True
        image_process(in_path, out_path)

    # Submit the processing task to an executor (thread or process)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_task, file_path, full_path, is_video)
        is_completed = future.result()

    if is_completed:
        processing_label.config(text="MEDIA HAS BEEN PROCESSED")

# Set up the main window
root = tk.Tk()
root.title("File Storage App")
root.geometry("800x600")  # Set window resolution

# Set up a frame to hold the file selection elements
file_frame = tk.Frame(root)
file_frame.pack(pady=20)

# Set up the read-only text box for displaying selected file name
selected_file_text = tk.Text(file_frame, height=1, width=80, state=tk.DISABLED)
selected_file_text.pack(side=tk.LEFT, padx=5)

# Set up the button for selecting files
button = tk.Button(file_frame, text="Select and Store Files", command=select_files)
button.pack(side=tk.LEFT)

# Set up a frame to hold the directory selection elements
dir_frame = tk.Frame(root)
dir_frame.pack(pady=20)

# Set up the read-only text box for displaying selected directory
selected_directory_text = tk.Text(dir_frame, height=1, width=80, state=tk.DISABLED)
selected_directory_text.pack(side=tk.LEFT, padx=5)

# Set up the button for selecting directory
dir_button = tk.Button(dir_frame, text="Select Directory", command=select_directory)
dir_button.pack(side=tk.LEFT)

# Set up the label for displaying selected file name
file_name_frame = tk.Frame(root)
file_name_frame.pack(pady=10, padx=20, fill=tk.X)

file_name_label = tk.Label(file_name_frame, text="File name:", anchor="e", width=12)
file_name_label.pack(side=tk.LEFT)

file_name_text = tk.Text(file_name_frame, height=1, width=80)
file_name_text.pack(side=tk.LEFT)

# Set up the processing label
processing_label = tk.Label(root, text="", font=("Helvetica", 16))
processing_label.pack(pady=20)

# Set up the button for processing media
process_button = tk.Button(root, text="Process Media", command=process_media)
process_button.pack(pady=20)

# Run the application
root.mainloop()
