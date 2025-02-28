import json
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Button
from path import Path
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

# ------------------ Load Configuration ------------------
with open('../data/config.json') as f:
    sample_config = json.load(f)

# Load word list
with open('../data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

# Get list of images
image_files = list(Path('../data').files('*.png'))
current_index = 0  # Track the current image index


# ------------------ Function to Process and Display Image ------------------
def process_and_display(index):
    """Processes the image at the given index and displays results."""
    global current_index
    if index >= len(image_files):
        print("✅ No more images to process.")
        return

    current_index = index
    img_filename = image_files[index]

    print(f'🔍 Processing {img_filename} ({index + 1}/{len(image_files)})')

    # Read input image
    img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
    scale = sample_config.get(img_filename.basename(), {}).get('scale', 1)
    margin = sample_config.get(img_filename.basename(), {}).get('margin', 0)

    # Perform Text Recognition
    read_lines = read_page(img,
                           detector_config=DetectorConfig(scale=scale, margin=margin),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                           reader_config=ReaderConfig(decoder='best_path', prefix_tree=prefix_tree))

    # Prepare extracted text
    extracted_text = "\n".join(" ".join(read_word.text for read_word in read_line) for read_line in read_lines)

    # Save text to a file
    text_filename = f"../data/{img_filename.stem}_best_path.txt"
    with open(text_filename, 'w', encoding='utf-8') as text_file:
        text_file.write(extracted_text)

    print(f"✅ Text saved to {text_filename}")

    # ------------------ Full Screen Display ------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))  # Full HD aspect ratio

    # Left Side: Display Input Image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("📷 Input Image", fontsize=16)
    axes[0].axis('off')

    # Overlay detected words with bounding boxes
    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            xs = [aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin]
            ys = [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin]
            axes[0].plot(xs, ys, c='r' if i % 2 else 'b')
            axes[0].text(aabb.xmin, aabb.ymin - 2, read_word.text, color='yellow', fontsize=10)

    # Right Side: Display Extracted Text
    axes[1].text(0, 1, extracted_text, fontsize=14, verticalalignment='top', wrap=True, color='black')
    axes[1].set_title("📝 Extracted Text", fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# ------------------ Next Button Function ------------------
def next_image():
    """Moves to the next image in the dataset."""
    global current_index
    if current_index + 1 < len(image_files):
        plt.close()  # Close current figure before moving to next
        process_and_display(current_index + 1)
    else:
        print("✅ All images processed!")


# ------------------ Full-Screen GUI ------------------
root = tk.Tk()
root.title("HTR Viewer")
root.geometry("300x100")  # Small window for buttons

# Create "Next" Button
next_button = Button(root, text="Next Image ▶", command=next_image, font=("Arial", 14))
next_button.pack(expand=True)

# Process First Image
process_and_display(0)

# Run GUI loop
root.mainloop()
