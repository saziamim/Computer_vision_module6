import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_evidence_frame(original_frame, flow, frame_idx, output_dir, step=16):
    """
    Save one evidence frame with optical flow arrows overlaid.
    """
    h, w = original_frame.shape[:2]
    vis = original_frame.copy()

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            end_point = (int(x + dx), int(y + dy))
            cv2.arrowedLine(vis, (x, y), end_point, (0, 255, 0), 1, tipLength=0.3)

    out_path = os.path.join(output_dir, f"evidence_frame_{frame_idx:04d}.png")
    cv2.imwrite(out_path, vis)


def compute_optical_flow(
    input_video_path,
    output_dir,
    max_seconds=30,
    resize_width=640,
    arrow_step=16,
    save_evidence_count=4
):
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frames_to_process = min(total_frames, int(max_seconds * fps))

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame from the video.")

    # Resize while preserving aspect ratio
    if resize_width is not None:
        h0, w0 = first_frame.shape[:2]
        scale = resize_width / w0
        resize_height = int(h0 * scale)
        first_frame = cv2.resize(first_frame, (resize_width, resize_height))
    else:
        resize_height, resize_width = first_frame.shape[:2]

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    # Output video paths
    hsv_video_path = os.path.join(output_dir, "optical_flow_hsv.mp4")
    arrow_video_path = os.path.join(output_dir, "optical_flow_arrows.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    hsv_writer = cv2.VideoWriter(hsv_video_path, fourcc, fps, (w, h))
    arrow_writer = cv2.VideoWriter(arrow_video_path, fourcc, fps, (w, h))

    avg_magnitudes = []
    evidence_dir = os.path.join(output_dir, "evidence_frames")
    ensure_dir(evidence_dir)

    # Select evidence frame indices roughly spread over the clip
    if frames_to_process > 1:
        evidence_indices = np.linspace(
            1, frames_to_process - 1, num=save_evidence_count, dtype=int
        )
        evidence_indices = set(evidence_indices.tolist())
    else:
        evidence_indices = set()

    processed_frame_count = 1

    while processed_frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        if resize_width is not None:
            frame = cv2.resize(frame, (w, h))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dense optical flow: Farneback
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Convert flow to magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Average magnitude for evidence/plot
        avg_mag = float(np.mean(mag))
        avg_magnitudes.append(avg_mag)

        # HSV visualization
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        hsv_writer.write(flow_hsv_bgr)

        # Arrow visualization
        arrow_frame = frame.copy()
        for y in range(0, h, arrow_step):
            for x in range(0, w, arrow_step):
                dx, dy = flow[y, x]
                end_point = (int(x + dx), int(y + dy))
                cv2.arrowedLine(
                    arrow_frame,
                    (x, y),
                    end_point,
                    (0, 255, 0),
                    1,
                    tipLength=0.3
                )
        arrow_writer.write(arrow_frame)

        # Save evidence frames
        if processed_frame_count in evidence_indices:
            save_evidence_frame(frame, flow, processed_frame_count, evidence_dir, step=arrow_step)

        prev_gray = gray
        processed_frame_count += 1

    cap.release()
    hsv_writer.release()
    arrow_writer.release()

    # Save motion magnitude plot
    plot_path = os.path.join(output_dir, "avg_motion_magnitude.png")
    plt.figure(figsize=(10, 4))
    plt.plot(avg_magnitudes)
    plt.xlabel("Frame index")
    plt.ylabel("Average optical flow magnitude")
    plt.title("Average Motion Magnitude Over Time")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Write summary text
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Part A - Optical Flow Summary\n")
        f.write("============================\n\n")
        f.write(f"Input video: {input_video_path}\n")
        f.write(f"Original duration: {duration:.2f} sec\n")
        f.write(f"Processed duration: {min(max_seconds, duration):.2f} sec\n")
        f.write(f"FPS used: {fps:.2f}\n")
        f.write(f"Frames processed: {processed_frame_count}\n\n")

        if len(avg_magnitudes) > 0:
            f.write(f"Minimum average motion magnitude: {np.min(avg_magnitudes):.4f}\n")
            f.write(f"Maximum average motion magnitude: {np.max(avg_magnitudes):.4f}\n")
            f.write(f"Mean average motion magnitude: {np.mean(avg_magnitudes):.4f}\n\n")

        f.write("Interpretation:\n")
        f.write("- Optical flow direction is represented by color in the HSV video.\n")
        f.write("- Optical flow magnitude is represented by brightness in the HSV video.\n")
        f.write("- In the arrow video, arrow direction shows motion direction.\n")
        f.write("- Larger arrows / brighter regions indicate stronger motion.\n")
        f.write("- Static regions usually have small or near-zero flow.\n")
        f.write("- If most of the frame moves similarly, that often indicates camera motion.\n")
        f.write("- If only a local region moves, that usually indicates object motion.\n")

    print("\nProcessing complete.")
    print(f"HSV optical flow video saved to: {hsv_video_path}")
    print(f"Arrow optical flow video saved to: {arrow_video_path}")
    print(f"Motion magnitude plot saved to: {plot_path}")
    print(f"Evidence frames saved to: {evidence_dir}")
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute optical flow for Part A")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--max_seconds", type=int, default=30)
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--arrow_step", type=int, default=16)
    parser.add_argument("--save_evidence_count", type=int, default=4)

    args = parser.parse_args()

    # automatically create output folder using video name
    video_name = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = os.path.join("outputs", video_name)

    compute_optical_flow(
        input_video_path=args.input,
        output_dir=output_dir,
        max_seconds=args.max_seconds,
        resize_width=args.resize_width,
        arrow_step=args.arrow_step,
        save_evidence_count=args.save_evidence_count
    )

if __name__ == "__main__":
    main()