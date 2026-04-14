"""
draw_bbox.py
Usage: python draw_bbox.py <image_path> <x> <y> <w> <h> <class_name> [--confidence <conf>]
Example: python draw_bbox.py image.jpg 163.9 155.3 88.6 211.8 "wine glass"
"""

import sys
import argparse
from PIL import Image, ImageDraw, ImageFont

def draw_bbox(image_path, x, y, w, h, class_name, confidence=None, output_path=None):
    # Open image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Convert coordinates to integers
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)

    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

    # Prepare label text
    label = class_name
    if confidence is not None:
        label += f" ({confidence:.4f})"

    # Draw label background and text
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Draw background rectangle for text
    draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="green")
    draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

    # Save or show
    if output_path is None:
        output_path = image_path.replace(".jpg", "_detection.jpg")
    img.save(output_path)
    print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding box on image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("x", type=float, help="X coordinate of top-left corner")
    parser.add_argument("y", type=float, help="Y coordinate of top-left corner")
    parser.add_argument("w", type=float, help="Width of bounding box")
    parser.add_argument("h", type=float, help="Height of bounding box")
    parser.add_argument("class_name", help="Class name (e.g., 'wine glass')")
    parser.add_argument("--confidence", type=float, help="Confidence score (optional)")
    parser.add_argument("--output", help="Output image path (default: input_detection.jpg)")

    args = parser.parse_args()
    draw_bbox(args.image, args.x, args.y, args.w, args.h, args.class_name, args.confidence, args.output)