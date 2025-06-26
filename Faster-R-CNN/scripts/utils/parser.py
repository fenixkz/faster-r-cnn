"""
A script that parses the names of the images and their labels into .csv file with additional information
"""

import os
import csv
import glob
import time
from PIL import Image 

# --- Configuration ---
BASE_PATH= "/home/fenixkz/.cache/kagglehub/datasets/fareselmenshawii/face-detection-dataset/versions/3" 

IMAGE_DIR = BASE_PATH + "/images"  
ANNOTATION_DIR = BASE_PATH + "/labels" 
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] 
# --- End Configuration ---

def scan_image_directory(image_dir):
    """
    Scans the image directory, gets dimensions, and creates a lookup map.
    Returns a dictionary: {base_filename: {'filename': full_filename, 'width': w, 'height': h}}
    """
    print(f"Scanning image directory: {image_dir} for extensions {IMAGE_EXTENSIONS}")
    start_time = time.time()
    image_files_map = {}
    skipped_image_reads = 0

    # Use glob to find all files matching the extensions directly in the directory
    all_files = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(image_dir, f"*{ext}")
        all_files.extend(glob.glob(pattern))

    # Process found files
    for img_path in all_files:
         if os.path.isfile(img_path): # Ensure it's a file
            full_filename = os.path.basename(img_path)
            base_filename = os.path.splitext(full_filename)[0]
            # Store the first encountered image for a given base name
            if base_filename not in image_files_map: # Process each base name only once
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_files_map[base_filename] = {
                            'filename': full_filename,
                            'width': width,
                            'height': height
                        }
                except Exception as e:
                    print(f"Warning: Could not read image '{full_filename}'. Skipping. Error: {e}")
                    skipped_image_reads += 1

    end_time = time.time()
    print(f"Found info for {len(image_files_map)} unique images in {end_time - start_time:.2f} seconds.")
    if skipped_image_reads > 0:
        print(f"Skipped reading {skipped_image_reads} image files due to errors.")
    if not image_files_map:
         print(f"Warning: No valid image files found or read in {image_dir} with specified extensions.")

    return image_files_map

def create_csv(image_dir, annotation_dir, output_csv):
    """
    Reads annotation files (.txt) and combines them into a single CSV file,
    using a pre-scanned list of images for efficiency.

    Args:
        image_dir (str): Path to the directory containing image files.
        annotation_dir (str): Path to the directory containing YOLO .txt files.
        output_csv (str): Path to save the resulting CSV file.
        image_extensions (list): List of possible image file extensions.
    """
    image_info_lookup = scan_image_directory(image_dir) # Scan image directory and get a lookup map

    # Initialize header for final .csv file
    header = ['image_name', 'image_width', 'image_height', 'class_id', 'x_center', 'y_center', 'width', 'height', 'format']
    all_rows = []

    # Find all .txt files in the annotation directory
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.txt'))

    if not annotation_files:
        print(f"Warning: No .txt annotation files found in {annotation_dir}")
        # Create an empty CSV with just the header if desired
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
        print(f"Empty CSV file with header created at {output_csv}")
        return

    print(f"Found {len(annotation_files)} annotation files. Processing...")
    start_process_time = time.time()

    processed_count = 0
    skipped_annotations = 0
    skipped_lines = 0

    for txt_file_path in annotation_files:
        base_filename = os.path.splitext(os.path.basename(txt_file_path))[0]

        image_info = image_info_lookup.get(base_filename) # Gets the dict {'filename': ..., 'width': ..., 'height': ...}

        if image_info is None:
            skipped_annotations += 1
            continue

        image_filename = image_info['filename']
        img_width = image_info['width']
        img_height = image_info['height']

        try:
            with open(txt_file_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                print(f"Info: Annotation file '{os.path.basename(txt_file_path)}' is empty.")
                pass # Handle empty files silently

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue # Skip empty lines within the file

                parts = line.split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # --- Append data including image dimensions ---
                        all_rows.append([
                            image_filename,
                            img_width,      
                            img_height,     
                            class_id,
                            x_center,
                            y_center,
                            width,
                            height,
                            "YOLO"
                        ])
                    except ValueError:
                        print(f"Warning: Invalid number format in {os.path.basename(txt_file_path)}, line {line_num + 1}: '{line}'. Skipping line.")
                        skipped_lines += 1
                else:
                    print(f"Warning: Incorrect number of values in {os.path.basename(txt_file_path)}, line {line_num + 1}: Expected 5, got {len(parts)} ('{line}'). Skipping line.")
                    skipped_lines += 1
            processed_count += 1

        except Exception as e:
            print(f"Error processing file {os.path.basename(txt_file_path)}: {e}")
            skipped_annotations += 1

    end_process_time = time.time()
    print(f"\nAnnotation processing took {end_process_time - start_process_time:.2f} seconds.")

    # Write all collected data to the CSV file
    try:
        start_write_time = time.time()
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header) # Write the header row
            writer.writerows(all_rows) # Write all the data rows
        end_write_time = time.time()
        print(f"CSV writing took {end_write_time - start_write_time:.2f} seconds.")

        print("\n--- Conversion Summary ---")
        print(f"Successfully processed {processed_count} annotation files.")
        print(f"Total bounding box rows added: {len(all_rows)}")
        if skipped_annotations > 0:
             print(f"Skipped {skipped_annotations} annotation files (e.g., due to missing images based on initial scan or read errors).")
        if skipped_lines > 0:
             print(f"Skipped {skipped_lines} lines within files (due to formatting issues).")
        print(f"CSV file created successfully at: {output_csv}")

    except IOError as e:
         print(f"\nError writing CSV file to {output_csv}: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during CSV writing: {e}")


if __name__ == "__main__":
    # Basic validation of paths
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Image directory not found at '{IMAGE_DIR}'")
    elif not os.path.isdir(ANNOTATION_DIR):
        print(f"Error: Annotation directory not found at '{ANNOTATION_DIR}'")
    else:
        # Ensure output directory exists
        output_dir = os.path.join(BASE_PATH, "csv")
        variants = ["train", 'val']
        for variant in variants:
            if output_dir and not os.path.exists(output_dir):
                print(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir)
            image_dir = os.path.join(IMAGE_DIR, variant)
            annotation_dir = os.path.join(ANNOTATION_DIR, variant)
            output_csv = os.path.join(output_dir, f"{variant}.csv")
            create_csv(image_dir, annotation_dir, output_csv)
