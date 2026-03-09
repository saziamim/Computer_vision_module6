import cv2
import numpy as np
import os
import argparse


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def bilinear_interpolate(image, x, y):
    """
    Bilinear interpolation for grayscale image at non-integer location (x, y).
    image: 2D numpy array
    x, y: float coordinates
    returns interpolated intensity
    """
    h, w = image.shape

    # keep inside valid range
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return None

    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    a = x - x0
    b = y - y0

    I00 = float(image[y0, x0])
    I10 = float(image[y0, x1])
    I01 = float(image[y1, x0])
    I11 = float(image[y1, x1])

    value = ((1 - a) * (1 - b) * I00 +
             a * (1 - b) * I10 +
             (1 - a) * b * I01 +
             a * b * I11)

    return value


def compute_derivatives(gray1, gray2):
    """
    Compute Ix, Iy, It for two consecutive grayscale frames.
    """
    gray1_f = gray1.astype(np.float32)
    gray2_f = gray2.astype(np.float32)

    # spatial derivatives from first frame
    Ix = cv2.Sobel(gray1_f, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(gray1_f, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    # temporal derivative
    It = gray2_f - gray1_f

    return Ix, Iy, It


def choose_tracking_points(gray, num_points=6):
    """
    Use goodFeaturesToTrack to automatically pick trackable points.
    If not enough points are found, fallback to a grid.
    """
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=num_points,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7
    )

    selected = []
    if pts is not None:
        for p in pts:
            x, y = p.ravel()
            selected.append((int(x), int(y)))

    # fallback grid if not enough points
    if len(selected) < num_points:
        h, w = gray.shape
        grid_points = [
            (w // 4, h // 4),
            (w // 2, h // 4),
            (3 * w // 4, h // 4),
            (w // 4, h // 2),
            (w // 2, h // 2),
            (3 * w // 4, h // 2),
            (w // 4, 3 * h // 4),
            (w // 2, 3 * h // 4),
            (3 * w // 4, 3 * h // 4),
        ]
        for pt in grid_points:
            if pt not in selected:
                selected.append(pt)
            if len(selected) >= num_points:
                break

    return selected[:num_points]


def validate_points(frame1, frame2, gray1, gray2, flow, Ix, Iy, It, points, output_dir):
    """
    Validate the theoretical tracking equation and save visualization.
    """
    vis1 = frame1.copy()
    vis2 = frame2.copy()

    report_lines = []
    report_lines.append("Part B - Motion Tracking Validation")
    report_lines.append("==================================")
    report_lines.append("")
    report_lines.append("Optical Flow Constraint Equation:")
    report_lines.append("Ix * u + Iy * v + It = 0")
    report_lines.append("")
    report_lines.append("For each selected point, the code reports:")
    report_lines.append("- original location (x, y)")
    report_lines.append("- estimated flow (u, v)")
    report_lines.append("- predicted location (x+u, y+v)")
    report_lines.append("- frame1 intensity at (x, y)")
    report_lines.append("- frame2 intensity at predicted location using bilinear interpolation")
    report_lines.append("- value of Ix*u + Iy*v + It (should be close to 0)")
    report_lines.append("")

    point_id = 1
    for (x, y) in points:
        if x < 1 or y < 1 or x >= gray1.shape[1] - 1 or y >= gray1.shape[0] - 1:
            continue

        u = float(flow[y, x, 0])
        v = float(flow[y, x, 1])

        pred_x = x + u
        pred_y = y + v

        i1 = float(gray1[y, x])
        i2_interp = bilinear_interpolate(gray2, pred_x, pred_y)

        if i2_interp is None:
            continue

        ix = float(Ix[y, x])
        iy = float(Iy[y, x])
        it = float(It[y, x])

        lhs = ix * u + iy * v + it
        intensity_diff = i2_interp - i1

        report_lines.append(f"Point {point_id}")
        report_lines.append(f"-------")
        report_lines.append(f"Original pixel location: ({x}, {y})")
        report_lines.append(f"Estimated flow (u, v): ({u:.4f}, {v:.4f})")
        report_lines.append(f"Predicted location: ({pred_x:.4f}, {pred_y:.4f})")
        report_lines.append(f"I1(x,y): {i1:.4f}")
        report_lines.append(f"I2(predicted location, bilinear): {i2_interp:.4f}")
        report_lines.append(f"Intensity difference I2-I1: {intensity_diff:.4f}")
        report_lines.append(f"Ix: {ix:.4f}, Iy: {iy:.4f}, It: {it:.4f}")
        report_lines.append(f"Ix*u + Iy*v + It = {lhs:.6f}")
        report_lines.append("Interpretation:")
        report_lines.append("- If Ix*u + Iy*v + It is close to 0, the theoretical optical flow equation is validated locally.")
        report_lines.append("- If I2 at predicted location is close to I1, brightness constancy is approximately satisfied.")
        report_lines.append("")

        # draw on frame1: original point
        cv2.circle(vis1, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            vis1, f"P{point_id}", (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )

        # draw arrow from original to predicted
        cv2.arrowedLine(
            vis1,
            (x, y),
            (int(round(pred_x)), int(round(pred_y))),
            (0, 255, 0),
            2,
            tipLength=0.25
        )

        # draw predicted point on frame2
        cv2.circle(vis2, (int(round(pred_x)), int(round(pred_y))), 5, (0, 255, 0), -1)
        cv2.putText(
            vis2, f"P{point_id}", (int(round(pred_x)) + 6, int(round(pred_y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )

        point_id += 1

    frame1_path = os.path.join(output_dir, "frame1_tracking.png")
    frame2_path = os.path.join(output_dir, "frame2_tracking.png")
    cv2.imwrite(frame1_path, vis1)
    cv2.imwrite(frame2_path, vis2)

    report_path = os.path.join(output_dir, "part_b_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    return frame1_path, frame2_path, report_path


def save_derivative_images(Ix, Iy, It, output_dir):
    """
    Save derivative maps for evidence.
    """
    def normalize_to_uint8(img):
        vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return vis.astype(np.uint8)

    ix_path = os.path.join(output_dir, "Ix.png")
    iy_path = os.path.join(output_dir, "Iy.png")
    it_path = os.path.join(output_dir, "It.png")

    cv2.imwrite(ix_path, normalize_to_uint8(Ix))
    cv2.imwrite(iy_path, normalize_to_uint8(Iy))
    cv2.imwrite(it_path, normalize_to_uint8(It))

    return ix_path, iy_path, it_path


def run_part_b(input_video_path, output_dir, resize_width=640, num_points=6):
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame.")

    ret, frame2 = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read second frame.")

    cap.release()

    # resize while keeping aspect ratio
    if resize_width is not None:
        h0, w0 = frame1.shape[:2]
        scale = resize_width / w0
        resize_height = int(h0 * scale)
        frame1 = cv2.resize(frame1, (resize_width, resize_height))
        frame2 = cv2.resize(frame2, (resize_width, resize_height))

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # dense optical flow between the two frames
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # derivatives
    Ix, Iy, It = compute_derivatives(gray1, gray2)

    # choose points to validate
    points = choose_tracking_points(gray1, num_points=num_points)

    # save raw frames too
    cv2.imwrite(os.path.join(output_dir, "frame1.png"), frame1)
    cv2.imwrite(os.path.join(output_dir, "frame2.png"), frame2)

    # save derivative images
    ix_path, iy_path, it_path = save_derivative_images(Ix, Iy, It, output_dir)

    # validate and save visualization
    frame1_path, frame2_path, report_path = validate_points(
        frame1, frame2, gray1, gray2, flow, Ix, Iy, It, points, output_dir
    )

    print("\nPart B processing complete.")
    print(f"Original frame 1 saved to: {os.path.join(output_dir, 'frame1.png')}")
    print(f"Original frame 2 saved to: {os.path.join(output_dir, 'frame2.png')}")
    print(f"Tracking visualization frame 1 saved to: {frame1_path}")
    print(f"Tracking visualization frame 2 saved to: {frame2_path}")
    print(f"Ix derivative image saved to: {ix_path}")
    print(f"Iy derivative image saved to: {iy_path}")
    print(f"It derivative image saved to: {it_path}")
    print(f"Text report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Part B: Motion tracking validation using two consecutive frames")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--resize_width", type=int, default=640, help="Resize width")
    parser.add_argument("--num_points", type=int, default=6, help="Number of validation points")

    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = os.path.join("outputs_part_b", video_name)

    run_part_b(
        input_video_path=args.input,
        output_dir=output_dir,
        resize_width=args.resize_width,
        num_points=args.num_points
    )


if __name__ == "__main__":
    main()