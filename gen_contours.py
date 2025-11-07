import cv2
import numpy as np
import svgwrite
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt

def extract_smooth_contours(image_path, output_path='smooth_contours.svg'):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Option 1: Multi-level contours (like topographic map)
    print("Extracting multi-level contours...")

    # Apply stronger Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Get intensity levels
    levels = np.percentile(blurred[blurred > 0], [20, 40, 60, 80])

    all_smooth_contours = []

    for i, level in enumerate(levels):
        # Use simple threshold for each level
        _, binary = cv2.threshold(blurred, level, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20000:  # Skip small noise
                continue

            # Smooth the contour
            smooth_contour = smooth_contour_spline(contour, smoothing_factor=0.1)
            all_smooth_contours.append((smooth_contour, level))

    # Save to SVG with different colors for different levels
    save_contours_to_svg(all_smooth_contours, output_path, img.shape)

    return all_smooth_contours

def smooth_contour_spline(contour, smoothing_factor=0.1):
    """Smooth contour using spline interpolation"""
    # Reshape contour
    contour = contour.reshape(-1, 2)

    # Close the contour by adding first point at end
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    # Calculate cumulative distance along contour
    distances = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)

    # Interpolate using splines
    from scipy import interpolate

    # Create periodic spline
    num_points = max(len(contour), 100)
    alpha = np.linspace(0, 1, num_points)

    # Fit spline
    try:
        tck, u = interpolate.splprep([contour[:, 0], contour[:, 1]],
                                     s=len(contour) * smoothing_factor,
                                     per=True)
        smooth_points = interpolate.splev(alpha, tck)
        smooth_contour = np.column_stack(smooth_points)
    except:
        # Fallback to simple smoothing if spline fails
        smooth_contour = contour

    return smooth_contour


def polygon_area(points: np.ndarray) -> float:
    """Return absolute area of a closed polygon given as Nx2 array."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def extract_marching_squares_contours(image_path, output_path='marching_squares.svg',
                                      levels=None, gaussian_kernel=(15, 15), gaussian_sigma=3,
                                      min_area=20000, smoothing_factor=0.08):
    """Extract contours using marching squares (skimage.find_contours)."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, gaussian_kernel, gaussian_sigma)

    if levels is None:
        mask = blurred > 0
        if not np.any(mask):
            return []
        levels = np.percentile(blurred[mask], [60, 80, 90, 95])

    marching_contours = []

    for level in levels:
        raw_contours = measure.find_contours(blurred, level=level)
        for contour in raw_contours:
            # skimage returns (row, col); flip to (x, y)
            contour_xy = contour[:, ::-1]
            if not np.array_equal(contour_xy[0], contour_xy[-1]):
                contour_xy = np.vstack([contour_xy, contour_xy[0]])

            if polygon_area(contour_xy) < min_area:
                continue

            smooth = smooth_contour_spline(contour_xy, smoothing_factor=smoothing_factor)
            marching_contours.append((smooth, level))

    if marching_contours:
        save_contours_to_svg(marching_contours, output_path, img.shape)

    return marching_contours

def save_contours_to_svg(contours_with_levels, output_path, image_shape,
                         fill_opacity=0.35, stroke_opacity=0.9, stroke_width=1.5,
                         draw_stroke=True):
    """Save contours to SVG with filled shapes (and optional stroke)."""
    height, width = image_shape[:2]
    dwg = svgwrite.Drawing(output_path, size=(width, height))

    # Nice 6-color set; repeats if needed
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7B267', '#CDB4DB']

    # Draw lower levels first so higher ones sit on top
    contours_with_levels = sorted(contours_with_levels, key=lambda x: x[1])

    for i, (contour, level) in enumerate(contours_with_levels):
        color = colors[i % len(colors)]
        pts = contour.tolist()
        if len(pts) < 3:
            continue

        # Build a simple closed path (straight segments). Beziers look nice for strokes
        # but can self-intersect when filled; straight segments are safer for fills.
        d = [f"M {pts[0][0]},{pts[0][1]}"]
        for j in range(1, len(pts)):
            d.append(f"L {pts[j][0]},{pts[j][1]}")
        d.append("Z")
        path_data = " ".join(d)

        path = dwg.path(
            d=path_data,
            fill=color,
            fill_opacity=fill_opacity,
            stroke=color if draw_stroke else 'none',
            stroke_opacity=stroke_opacity,
            stroke_width=stroke_width
        )

        # Helps when there are holes; keeps visual sane without hierarchy bookkeeping
        path.update({'fill-rule': 'evenodd'})

        dwg.add(path)

    dwg.save()
    print(f"Saved smooth filled contours to {output_path}")


# def save_contours_to_svg(contours_with_levels, output_path, image_shape):
#     """Save contours to SVG with different colors"""
#     height, width = image_shape[:2]
#     dwg = svgwrite.Drawing(output_path, size=(width, height))

#     # Color map for different levels
#     colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

#     for i, (contour, level) in enumerate(contours_with_levels):
#         color = colors[i % len(colors)]
#         points = contour.tolist()

#         # Create smooth path
#         if len(points) > 2:
#             path_data = f"M {points[0][0]},{points[0][1]} "

#             # Use quadratic bezier curves for smoothness
#             for j in range(1, len(points) - 1):
#                 ctrl_x = points[j][0]
#                 ctrl_y = points[j][1]
#                 end_x = (points[j][0] + points[j + 1][0]) / 2
#                 end_y = (points[j][1] + points[j + 1][1]) / 2
#                 path_data += f"Q {ctrl_x},{ctrl_y} {end_x},{end_y} "

#             # Close path
#             path_data += "Z"

#             path = dwg.path(d=path_data,
#                           fill='none',
#                           stroke=color,
#                           stroke_width=2,
#                           opacity=0.8)
#             dwg.add(path)

#     dwg.save()
#     print(f"Saved smooth contours to {output_path}")

def extract_edge_based_contours(image_path, output_path='edge_contours.svg'):
    """Alternative approach using Canny edge detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Apply Canny edge detection
    edges = cv2.Canny(denoised, 50, 150)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and smooth
    smooth_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 20000:
            smooth = smooth_contour_spline(contour, smoothing_factor=0.05)
            smooth_contours.append((smooth, 100))

    save_contours_to_svg(smooth_contours, output_path, img.shape)
    return smooth_contours

# Main execution
if __name__ == "__main__":
    # Try both approaches
    print("Method 1: Multi-level contours")
    contours1 = extract_smooth_contours('spectrogram.png', 'smooth_contours.svg')

    print("\nMethod 2: Edge-based contours")
    contours2 = extract_edge_based_contours('spectrogram.png', 'edge_contours.svg')

    print("\nMethod 3: Marching-squares contours")
    contours3 = extract_marching_squares_contours('spectrogram.png', 'marching_squares_contours.svg')

    # Optional: Create a preview image
    img = cv2.imread('spectrogram.png')
    preview = img.copy()

    # Draw smooth contours on preview
    for contour, level in contours3[:10] if contours3 else contours1[:10]:
        cv2.drawContours(preview, [contour.astype(int)], -1, (0, 255, 0), 2)

    cv2.imwrite('contour_preview.png', preview)
    print("\nPreview saved to contour_preview.png")