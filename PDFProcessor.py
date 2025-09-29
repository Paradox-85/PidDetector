import sys
import os.path
from matplotlib import patches
from ultralytics import YOLO

def format_label(texts):
    if not texts:
        return "N/A"
    
    if len(texts) == 3:
        return f"{texts[0]}{texts[1]}-{texts[2]}"
    else:
        return "-".join(texts)

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Update poppler path for PyInstaller
poppler_path = r"C:\poppler\poppler-24.08.0\Library\bin"
padding = 15


def get_model() -> YOLO:
    # Check if running as exe
    if hasattr(sys, '_MEIPASS'):
        # Running as exe - look for model in same directory as exe
        exe_dir = os.path.dirname(sys.executable)
        model_path = os.path.join(exe_dir, 'models', 'best.pt')
        # If not found, try current directory
        if not os.path.exists(model_path):
            model_path = os.path.join(exe_dir, 'best.pt')
    else:
        # Running as script - use original path
        model_path = get_resource_path(os.path.join('runs', 'detect', 'train27', 'weights', 'best.pt'))

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please place best.pt in the same folder as the executable.")

    return YOLO(model_path)


def apply_padding_to_crop(x1, y1, x2, y2, image_height, image_width, padding=15):
    """
    Apply padding to crop coordinates while ensuring they stay within image boundaries

    Args:
        x1, y1, x2, y2: Original bounding box coordinates
        image_height, image_width: Dimensions of the source image
        padding: Padding to add around the crop

    Returns:
        Padded coordinates (x1_pad, y1_pad, x2_pad, y2_pad)
    """
    x1_pad = max(0, int(x1) - padding)
    y1_pad = max(0, int(y1) - padding)
    x2_pad = min(image_width, int(x2) + padding)
    y2_pad = min(image_height, int(y2) + padding)

    return x1_pad, y1_pad, x2_pad, y2_pad


def plot_detections_cv2(image, boxes, model, font_scale=0.7, thickness=2, show_conf=True):
    """
    Plot detections using OpenCV with customizable font and line thickness
    """
    img_display = image.copy()
    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        color = colors[cls_id % len(colors)]

        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, thickness)

        label = model.names[cls_id]
        if show_conf:
            label = f'{label} {conf:.2f}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(img_display,
                      (x1, y1 - text_height - 10),
                      (x1 + text_width + 5, y1),
                      color, -1)

        cv2.putText(img_display, label, (x1 + 2, y1 - 5),
                    font, font_scale, (255, 255, 255), thickness)

    return img_display


def plot_detections_matplotlib(image, boxes, model, figsize=(15, 10),
                               font_size=10, linewidth=2, show_conf=True):
    """
    Plot detections using Matplotlib with customizable font and line thickness
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'purple', 'orange']

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        cls_id = int(cls_id)
        color = colors[cls_id % len(colors)]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label = model.names[cls_id]
        if show_conf:
            label = f'{label} {conf:.2f}'

        ax.text(x1, y1 - 5, label, fontsize=font_size, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                          edgecolor='none', alpha=0.7))

    plt.tight_layout()
    return fig


from concurrent.futures import ThreadPoolExecutor, as_completed
from paddlex import create_pipeline
import os
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import matplotlib.pyplot as plt


def get_data_from_pdf(pdf_path: str, progress_callback,
                      visualize='matplotlib', font_scale=0.7, thickness=2,
                      font_size=10, linewidth=2, show_conf=True,
                      save_path=None, crop_padding=15):
    """
    Process PDF, visualize detections, and run OCR with robust file handling.
    """
    try:
        model = get_model()
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        image = np.array(pages[0])
        image_height, image_width = image.shape[:2]

        results = model(image, conf=0.10)
        boxes = results[0].boxes.data
        total = len(boxes)

        # Visualization
        if visualize == 'cv2':
            img_with_boxes = plot_detections_cv2(image, boxes, model,
                                                 font_scale=font_scale,
                                                 thickness=thickness,
                                                 show_conf=show_conf)
            cv2.imshow('Detections', img_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if save_path:
                cv2.imwrite(save_path, img_with_boxes)

        elif visualize == 'matplotlib':
            fig = plot_detections_matplotlib(image, boxes, model,
                                             font_size=font_size,
                                             linewidth=linewidth,
                                             show_conf=show_conf)
            plt.show()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

        # OCR Pipeline with robust file handling
        import threading
        import gc
        import time
        import tempfile
        import uuid
        from pathlib import Path

        # Create OCR pipeline ONCE at the beginning
        print("Initializing OCR pipeline...")
        ocr = create_pipeline(pipeline="OCR")
        print("OCR pipeline initialized successfully!")

        # Global lock for OCR operations
        ocr_global_lock = threading.Lock()

        def safe_create_temp_file(box_index):
            """Create a temporary file with multiple fallback strategies"""
            max_attempts = 3
            temp_file_path = None

            for attempt in range(max_attempts):
                try:
                    # Strategy 1: Use tempfile with custom naming
                    if attempt == 0:
                        temp_dir = tempfile.gettempdir()
                        unique_id = str(uuid.uuid4())[:8]
                        timestamp = str(int(time.time() * 1000))
                        filename = f"ocr_{box_index}_{timestamp}_{unique_id}.png"
                        temp_file_path = os.path.join(temp_dir, filename)

                    # Strategy 2: Use current directory as fallback
                    elif attempt == 1:
                        unique_id = str(uuid.uuid4())[:8]
                        timestamp = str(int(time.time() * 1000))
                        filename = f"temp_ocr_{box_index}_{timestamp}_{unique_id}.png"
                        temp_file_path = os.path.join(os.getcwd(), filename)

                    # Strategy 3: Use system temp with shorter path
                    else:
                        unique_id = str(uuid.uuid4())[:6]
                        filename = f"ocr_{unique_id}.png"
                        temp_file_path = os.path.join(tempfile.gettempdir(), filename)

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                    # Test if we can create the file
                    with open(temp_file_path, 'wb') as test_file:
                        test_file.write(b'test')

                    # Clean up test file
                    os.unlink(temp_file_path)

                    return temp_file_path

                except Exception as e:
                    print(f"Temp file creation attempt {attempt + 1} failed: {e}")
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                    continue

            raise Exception("Failed to create temporary file after all attempts")

        def safe_write_and_verify(image_data, file_path, max_retries=3):
            """Safely write image and verify it exists"""
            for retry in range(max_retries):
                try:
                    # Write the image
                    success = cv2.imwrite(file_path, image_data)
                    if not success:
                        raise Exception(f"cv2.imwrite returned False for {file_path}")

                    # Wait for file system to sync
                    time.sleep(0.05)  # Increased wait time

                    # Verify file exists and has content
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        # Additional verification - try to read the file
                        test_img = cv2.imread(file_path)
                        if test_img is not None:
                            return True
                        else:
                            raise Exception("File exists but cannot be read by cv2")
                    else:
                        raise Exception(
                            f"File verification failed: exists={os.path.exists(file_path)}, size={os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")

                except Exception as e:
                    print(f"Write attempt {retry + 1} failed for {file_path}: {e}")
                    if retry < max_retries - 1:
                        time.sleep(0.1 * (retry + 1))  # Exponential backoff
                    continue

            return False

        def process_box_safe(i, box):
            """Process single box with enhanced error handling"""
            x1, y1, x2, y2, conf, cls_id = box.tolist()
            shape_type = model.names[int(cls_id)]

            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            if cropped.size == 0:
                return {
                    "Shape": shape_type,
                    "Label": "N/A",
                    "X": int(x1),
                    "Y": int(y1),
                    "Width": int(x2 - x1),
                    "Height": int(y2 - y1),
                    "PDF Name": os.path.basename(pdf_path)
                }

            temp_file_path = None
            try:
                # Image preprocessing
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                denoised = cv2.fastNlMeansDenoising(resized, h=30)
                sharpened = cv2.filter2D(denoised, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
                thresh = cv2.adaptiveThreshold(sharpened, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 31, 2)

                # Thread-safe OCR processing
                with ocr_global_lock:
                    try:
                        # Create safe temporary file
                        temp_file_path = safe_create_temp_file(i)

                        # Write and verify image file
                        if not safe_write_and_verify(thresh, temp_file_path):
                            raise Exception(f"Failed to create valid temporary file: {temp_file_path}")

                        print(f"Processing box {i}: {temp_file_path} (size: {os.path.getsize(temp_file_path)} bytes)")

                        # Run OCR using the shared OCR instance
                        results_ocr = ocr.predict(temp_file_path)

                    except Exception as ocr_error:
                        print(f"OCR prediction failed for box {i}: {str(ocr_error)}")
                        results_ocr = None

                # Extract text results
                texts = []
                if results_ocr and 'ocr' in results_ocr:
                    for item in results_ocr['ocr']:
                        if 'text' in item and item['text'].strip():
                            texts.append(item['text'].strip())

                # generate final tag name
                final_label = format_label(texts)

            except Exception as e:
                print(f"OCR processing error on box {i}: {e}")
                final_label = "N/A"
            finally:
                # Clean up temp file with multiple attempts
                if temp_file_path:
                    for cleanup_attempt in range(3):
                        try:
                            if os.path.exists(temp_file_path):
                                os.unlink(temp_file_path)
                                break
                        except Exception as cleanup_error:
                            if cleanup_attempt == 2:  # Last attempt
                                print(f"Warning: Could not delete {temp_file_path}: {cleanup_error}")
                            else:
                                time.sleep(0.1)

            # Update progress
            if progress_callback:
                progress_callback(int((i + 1) / total * 100))

            return {
                "Shape": shape_type,
                "Label": final_label,
                "X": int(x1),
                "Y": int(y1),
                "Width": int(x2 - x1),
                "Height": int(y2 - y1),
                "PDF Name": os.path.basename(pdf_path)
            }

        # Process boxes sequentially for maximum stability
        print(f"Processing {total} boxes sequentially with enhanced error handling...")
        data = []

        try:
            for i, box in enumerate(boxes):
                try:
                    result = process_box_safe(i, box)
                    data.append(result)

                    # Periodic cleanup (but don't delete the main OCR instance)
                    if i % 5 == 0:
                        gc.collect()
                        time.sleep(0.1)  # Brief pause every 5 boxes

                except Exception as box_error:
                    print(f"Fatal error processing box {i}: {box_error}")
                    # Add failed box with N/A label
                    x1, y1, x2, y2, conf, cls_id = box.tolist()
                    data.append({
                        "Shape": model.names[int(cls_id)],
                        "Label": "N/A",
                        "X": int(x1),
                        "Y": int(y1),
                        "Width": int(x2 - x1),
                        "Height": int(y2 - y1),
                        "PDF Name": os.path.basename(pdf_path)
                    })
        finally:
            # Clean up the main OCR instance at the end
            try:
                del ocr
                gc.collect()
                print("OCR pipeline cleaned up successfully!")
            except:
                pass

        return pd.DataFrame(data)

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")


# Alternative: In-memory processing version (if your OCR supports it)
def get_data_from_pdf_memory(pdf_path: str, progress_callback,
                             visualize='matplotlib', font_scale=0.7, thickness=2,
                             font_size=10, linewidth=2, show_conf=True,
                             save_path=None, crop_padding=15):
    """
    Process PDF using in-memory approach to avoid file system issues entirely.
    Note: This requires your OCR pipeline to support PIL Image or numpy array input.
    """
    try:
        model = get_model()
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        image = np.array(pages[0])
        image_height, image_width = image.shape[:2]

        results = model(image, conf=0.10)
        boxes = results[0].boxes.data
        total = len(boxes)

        # Visualization code (same as before)
        if visualize == 'matplotlib':
            fig = plot_detections_matplotlib(image, boxes, model,
                                             font_size=font_size,
                                             linewidth=linewidth,
                                             show_conf=show_conf)
            plt.show()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

        # Try to use OCR without temporary files
        import threading
        import gc
        from PIL import Image

        ocr_global_lock = threading.Lock()

        def process_box_memory(i, box):
            x1, y1, x2, y2, conf, cls_id = box.tolist()
            shape_type = model.names[int(cls_id)]

            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            if cropped.size == 0:
                return {
                    "Shape": shape_type,
                    "Label": "N/A",
                    "X": int(x1),
                    "Y": int(y1),
                    "Width": int(x2 - x1),
                    "Height": int(y2 - y1),
                    "PDF Name": os.path.basename(pdf_path)
                }

            try:
                # Process image
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                denoised = cv2.fastNlMeansDenoising(resized, h=30)
                sharpened = cv2.filter2D(denoised, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
                thresh = cv2.adaptiveThreshold(sharpened, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 31, 2)

                with ocr_global_lock:
                    ocr_local = create_pipeline(pipeline="OCR")

                    try:
                        # Convert to PIL Image (if your OCR supports it)
                        pil_image = Image.fromarray(thresh)

                        # Try direct image processing (modify based on your OCR API)
                        # results_ocr = ocr_local.predict(pil_image)  # If supported
                        # OR fallback to base64 encoding if supported
                        # import base64
                        # import io
                        # buffer = io.BytesIO()
                        # pil_image.save(buffer, format='PNG')
                        # img_str = base64.b64encode(buffer.getvalue()).decode()
                        # results_ocr = ocr_local.predict_base64(img_str)  # If supported

                        # If none of the above work, use a more reliable temp file method
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            pil_image.save(tmp.name)
                            tmp.flush()
                            os.fsync(tmp.fileno())  # Force write to disk
                            results_ocr = ocr_local.predict(tmp.name)
                            os.unlink(tmp.name)

                    finally:
                        del ocr_local
                        gc.collect()

                texts = []
                if results_ocr and 'ocr' in results_ocr:
                    for item in results_ocr['ocr']:
                        if 'text' in item and item['text'].strip():
                            texts.append(item['text'].strip())

                final_label = format_label(texts)

            except Exception as e:
                print(f"OCR processing error on box {i}: {e}")
                final_label = "N/A"

            if progress_callback:
                progress_callback(int((i + 1) / total * 100))

            return {
                "Shape": shape_type,
                "Label": final_label,
                "X": int(x1),
                "Y": int(y1),
                "Width": int(x2 - x1),
                "Height": int(y2 - y1),
                "PDF Name": os.path.basename(pdf_path)
            }

        # Process sequentially
        data = []
        for i, box in enumerate(boxes):
            data.append(process_box_memory(i, box))

        return pd.DataFrame(data)

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")


import easyocr
import cv2
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def get_data_from_pdf_easyocr(pdf_path: str, progress_callback,
                              visualize='matplotlib', font_scale=0.7, thickness=2,
                              font_size=10, linewidth=2, show_conf=True,
                              save_path=None, crop_padding=15):
    """
    Process PDF using EasyOCR - Highly recommended alternative
    """
    try:
        model = get_model()
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        image = np.array(pages[0])
        image_height, image_width = image.shape[:2]

        results = model(image, conf=0.10)
        boxes = results[0].boxes.data
        total = len(boxes)

        # Visualization (same as before)
        if visualize == 'matplotlib':
            fig = plot_detections_matplotlib(image, boxes, model,
                                             font_size=font_size,
                                             linewidth=linewidth,
                                             show_conf=show_conf)
            plt.show()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

        # Initialize EasyOCR reader once
        print("Initializing EasyOCR...")
        reader = easyocr.Reader(['en'])  # Add more languages as needed: ['en', 'ch_sim', 'fr']
        print("EasyOCR initialized successfully!")

        ocr_lock = threading.Lock()

        def process_box_easyocr(i, box):
            x1, y1, x2, y2, conf, cls_id = box.tolist()
            shape_type = model.names[int(cls_id)]

            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            if cropped.size == 0:
                return {
                    "Shape": shape_type,
                    "Label": "N/A",
                    "X": int(x1),
                    "Y": int(y1),
                    "Width": int(x2 - x1),
                    "Height": int(y2 - y1),
                    "PDF Name": os.path.basename(pdf_path)
                }

            try:
                # Image preprocessing
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                # EasyOCR works directly with numpy arrays - no temp files needed!
                with ocr_lock:
                    results_ocr = reader.readtext(resized)

                # Extract text from EasyOCR results
                texts = []
                for (bbox, text, confidence) in results_ocr:
                    if confidence > 0.5 and text.strip():  # Filter by confidence
                        texts.append(text.strip())

                final_label = format_label(texts)

            except Exception as e:
                print(f"EasyOCR processing error on box {i}: {e}")
                final_label = "N/A"

            if progress_callback:
                progress_callback(int((i + 1) / total * 100))

            return {
                "Shape": shape_type,
                "Label": final_label,
                "X": int(x1),
                "Y": int(y1),
                "Width": int(x2 - x1),
                "Height": int(y2 - y1),
                "PDF Name": os.path.basename(pdf_path)
            }

        # Process boxes (can use parallel processing safely with EasyOCR)
        print(f"Processing {total} boxes with EasyOCR...")
        data = []

        # Sequential processing (safer)
        for i, box in enumerate(boxes):
            result = process_box_easyocr(i, box)
            data.append(result)

            if progress_callback:
                progress_callback(int((i + 1) / total * 100))

        return pd.DataFrame(data)

    except Exception as e:
        raise Exception(f"Error processing PDF with EasyOCR: {str(e)}")