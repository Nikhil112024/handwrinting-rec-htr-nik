import json
import cv2
import matplotlib.pyplot as plt
from path import Path

from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

# Load configuration
with open('../data/config.json') as f:
    sample_config = json.load(f)

# Load word list
with open('../data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

# Process images and save results
for decoder in ['best_path', 'word_beam_search']:
    for img_filename in Path('../data').files('*.png'):
        print(f'Reading file {img_filename} with decoder {decoder}')

        # Read image
        img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
        scale = sample_config.get(img_filename.basename(), {}).get('scale', 1)
        margin = sample_config.get(img_filename.basename(), {}).get('margin', 0)

        read_lines = read_page(img,
                               detector_config=DetectorConfig(scale=scale, margin=margin),
                               line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                               reader_config=ReaderConfig(decoder=decoder, prefix_tree=prefix_tree))

        # Prepare text output
        extracted_text = "\n".join(" ".join(read_word.text for read_word in read_line) for read_line in read_lines)

        # Save text to a file
        text_filename = f"../data/{img_filename.stem}_{decoder}.txt"
        with open(text_filename, 'w', encoding='utf-8') as text_file:
            text_file.write(extracted_text)

        print(f'Text saved to {text_filename}')

        # Plot image with detected text overlay
        plt.figure(f'Image: {img_filename} Decoder: {decoder}')
        plt.imshow(img, cmap='gray')
        for i, read_line in enumerate(read_lines):
            for read_word in read_line:
                aabb = read_word.aabb
                xs = [aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin]
                ys = [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin]
                plt.plot(xs, ys, c='r' if i % 2 else 'b')
                plt.text(aabb.xmin, aabb.ymin - 2, read_word.text)

plt.show()
