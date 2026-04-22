import cv2
import numpy as np
import os
import argparse
import json
from pathlib import Path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_images(image_paths, resize_width=None):
    images = []
    loaded_paths = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] Could not read image: {path}")
            continue

        if resize_width is not None:
            h, w = img.shape[:2]
            scale = resize_width / float(w)
            new_h = int(h * scale)
            img = cv2.resize(img, (resize_width, new_h))

        images.append(img)
        loaded_paths.append(path)

    if len(images) < 2:
        raise ValueError("Need at least 2 valid images. For this assignment, use 4 images.")

    return images, loaded_paths


def detect_and_describe(image, max_features=1500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return gray, keypoints, descriptors


def match_features(desc1, desc2, ratio_test=0.75):
    if desc1 is None or desc2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)

    return good


def draw_keypoints(image, keypoints, output_path):
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(output_path, vis)


def draw_matches(img1, kp1, img2, kp2, matches, output_path, max_draw=80):
    matches = sorted(matches, key=lambda m: m.distance)
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:max_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, vis)


def compute_homography(kp_ref, kp_cur, matches, ransac_thresh=4.0):
    if len(matches) < 4:
        return None, None, None, None

    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    cur_pts = np.float32([kp_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ref_pts, cur_pts, cv2.RANSAC, ransac_thresh)
    return H, mask, ref_pts, cur_pts


def save_inlier_matches(img_ref, kp_ref, img_cur, kp_cur, matches, mask, output_path, max_draw=80):
    if mask is None:
        draw_matches(img_ref, kp_ref, img_cur, kp_cur, matches, output_path, max_draw=max_draw)
        return

    mask = mask.ravel().tolist()
    inlier_matches = [m for m, keep in zip(matches, mask) if keep]
    draw_matches(img_ref, kp_ref, img_cur, kp_cur, inlier_matches, output_path, max_draw=max_draw)


def select_reference_corners_manual_or_full_image(image, manual_corners=None):
    """
    If manual corners are given, use them.
    Otherwise, use the full image boundary as a simple default.
    For best results in the assignment, provide the 4 corners of the planar object manually.
    """
    h, w = image.shape[:2]

    if manual_corners is not None:
        pts = np.array(manual_corners, dtype=np.float32).reshape(-1, 1, 2)
    else:
        pts = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32
        ).reshape(-1, 1, 2)

    return pts


def draw_projected_boundary(image, projected_pts, output_path, color=(0, 255, 0), thickness=3):
    vis = image.copy()
    pts = np.int32(projected_pts)
    cv2.polylines(vis, [pts], True, color, thickness)
    cv2.imwrite(output_path, vis)


def warp_to_reference(current_image, H_cur_from_ref, ref_size):
    """
    H_cur_from_ref maps reference -> current.
    To warp current image back to reference plane, use inverse(H).
    """
    h_ref, w_ref = ref_size[:2]
    H_ref_from_cur = np.linalg.inv(H_cur_from_ref)
    warped = cv2.warpPerspective(current_image, H_ref_from_cur, (w_ref, h_ref))
    return warped


def blend_images(base, overlay, alpha=0.5):
    if base.shape != overlay.shape:
        raise ValueError("Images to blend must have the same size.")
    return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)


def save_homography_text(H, output_path):
    with open(output_path, "w") as f:
        f.write("Homography matrix H:\n")
        f.write("====================\n\n")
        f.write(str(H))
        f.write("\n\n")
        f.write("This matrix maps points from the reference image to the current image:\n")
        f.write("x' = Hx\n")


def create_camera_info_template(image_paths, output_path):
    template = []
    for idx, path in enumerate(image_paths, start=1):
        template.append({
            "image_id": idx,
            "file": os.path.basename(path),
            "camera_position_description": f"View {idx}: describe where camera was placed",
            "distance_from_object": "fill manually",
            "tilt_angle": "fill manually",
            "camera_device": "fill manually",
            "focal_length_if_known": "fill manually",
            "notes": "fill manually"
        })

    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)


def create_summary_report(
    output_path,
    image_paths,
    homography_results,
    ref_corners,
    note_manual_corners_used
):
    with open(output_path, "w") as f:
        f.write("Part B: Planar Structure from Motion / Homography-based Reconstruction\n")
        f.write("=====================================================================\n\n")

        f.write("Input images:\n")
        for i, p in enumerate(image_paths, start=1):
            f.write(f"  {i}. {p}\n")
        f.write("\n")

        f.write("Method overview:\n")
        f.write("- A planar object was captured from four viewpoints.\n")
        f.write("- ORB features were detected in each image.\n")
        f.write("- Feature matches were computed between the reference image and each other image.\n")
        f.write("- A homography H was estimated using RANSAC.\n")
        f.write("- The planar boundary from the reference image was projected into the other images.\n")
        f.write("- Each image was warped back to the reference plane to approximate a common planar reconstruction.\n\n")

        f.write("Mathematical model:\n")
        f.write("For planar scenes, corresponding points satisfy:\n")
        f.write("    x' = Hx\n")
        f.write("where H is a 3x3 homography matrix.\n\n")

        if note_manual_corners_used:
            f.write("Boundary selection:\n")
            f.write("- Manual object corners were used in the reference image.\n\n")
        else:
            f.write("Boundary selection:\n")
            f.write("- The full reference image boundary was used as a default.\n")
            f.write("- For stronger assignment evidence, manually provide the true object corners.\n\n")

        f.write("Reference boundary corners:\n")
        ref_corners_2d = ref_corners.reshape(-1, 2)
        for idx, (x, y) in enumerate(ref_corners_2d, start=1):
            f.write(f"  Corner {idx}: ({x:.2f}, {y:.2f})\n")
        f.write("\n")

        for result in homography_results:
            f.write(f"Pair: reference -> image {result['image_index']}\n")
            f.write("----------------------------------------\n")
            f.write(f"Total good matches: {result['num_matches']}\n")
            f.write(f"RANSAC inliers: {result['num_inliers']}\n")
            if result["H"] is not None:
                f.write("Homography matrix:\n")
                f.write(np.array2string(result["H"], precision=4, suppress_small=True))
                f.write("\n")
            else:
                f.write("Homography could not be estimated.\n")
            f.write("\n")

        f.write("Interpretation:\n")
        f.write("- If the projected boundary aligns with the planar object in each view, the planar mapping is validated.\n")
        f.write("- If the warped images align well in the reference plane, this supports reconstruction of the planar surface.\n")
        f.write("- Small errors are expected due to imperfect matches, camera noise, and perspective effects.\n")


def parse_manual_corners(corners_str):
    """
    Expected format:
    "x1,y1;x2,y2;x3,y3;x4,y4"
    Example:
    "120,80;520,90;530,430;110,420"
    """
    if corners_str is None:
        return None

    pts = []
    parts = corners_str.strip().split(";")
    if len(parts) != 4:
        raise ValueError("manual_corners must contain exactly 4 points.")

    for p in parts:
        x_str, y_str = p.split(",")
        pts.append([float(x_str), float(y_str)])

    return pts


def main():
    parser = argparse.ArgumentParser(
        description="Part B: Planar structure-from-motion style reconstruction using 4 viewpoints"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to input images. Use 4 images of the same planar object."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="part_b_outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=800,
        help="Resize width for all images"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=1500,
        help="Maximum ORB features"
    )
    parser.add_argument(
        "--ratio_test",
        type=float,
        default=0.75,
        help="Lowe ratio test threshold"
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=4.0,
        help="RANSAC reprojection threshold"
    )
    parser.add_argument(
        "--manual_corners",
        type=str,
        default=None,
        help='Optional reference object corners as "x1,y1;x2,y2;x3,y3;x4,y4"'
    )

    args = parser.parse_args()

    if len(args.images) < 4:
        print("[WARNING] The assignment asks for 4 viewpoints. You provided fewer than 4 images.")

    ensure_dir(args.output_dir)

    images, image_paths = load_images(args.images, resize_width=args.resize_width)

    # Save resized originals
    originals_dir = os.path.join(args.output_dir, "01_resized_inputs")
    ensure_dir(originals_dir)
    for idx, img in enumerate(images, start=1):
        cv2.imwrite(os.path.join(originals_dir, f"image_{idx}.png"), img)

    # Reference image is the first image
    ref_img = images[0]
    ref_gray, ref_kp, ref_desc = detect_and_describe(ref_img, max_features=args.max_features)

    # Save reference keypoints
    features_dir = os.path.join(args.output_dir, "02_features")
    ensure_dir(features_dir)
    draw_keypoints(ref_img, ref_kp, os.path.join(features_dir, "ref_keypoints.png"))

    # Parse manual corners if given
    manual_corners = parse_manual_corners(args.manual_corners)
    ref_corners = select_reference_corners_manual_or_full_image(ref_img, manual_corners)

    # Draw reference corners / polygon
    ref_boundary_vis = ref_img.copy()
    cv2.polylines(ref_boundary_vis, [np.int32(ref_corners)], True, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(args.output_dir, "reference_boundary.png"), ref_boundary_vis)

    matches_dir = os.path.join(args.output_dir, "03_matches")
    ensure_dir(matches_dir)

    proj_dir = os.path.join(args.output_dir, "04_projected_boundaries")
    ensure_dir(proj_dir)

    warp_dir = os.path.join(args.output_dir, "05_warped_to_reference")
    ensure_dir(warp_dir)

    blend_dir = os.path.join(args.output_dir, "06_blended_reconstruction")
    ensure_dir(blend_dir)

    matrices_dir = os.path.join(args.output_dir, "07_homography_matrices")
    ensure_dir(matrices_dir)

    homography_results = []
    blended = ref_img.copy()

    # Save reference warped image too
    cv2.imwrite(os.path.join(warp_dir, "warped_ref.png"), ref_img)

    for i in range(1, len(images)):
        cur_img = images[i]
        cur_gray, cur_kp, cur_desc = detect_and_describe(cur_img, max_features=args.max_features)

        draw_keypoints(
            cur_img,
            cur_kp,
            os.path.join(features_dir, f"image_{i+1}_keypoints.png")
        )

        good_matches = match_features(ref_desc, cur_desc, ratio_test=args.ratio_test)

        draw_matches(
            ref_img,
            ref_kp,
            cur_img,
            cur_kp,
            good_matches,
            os.path.join(matches_dir, f"matches_ref_to_{i+1}_all.png")
        )

        H, mask, ref_pts, cur_pts = compute_homography(
            ref_kp, cur_kp, good_matches, ransac_thresh=args.ransac_thresh
        )

        num_inliers = int(mask.sum()) if mask is not None else 0

        save_inlier_matches(
            ref_img,
            ref_kp,
            cur_img,
            cur_kp,
            good_matches,
            mask,
            os.path.join(matches_dir, f"matches_ref_to_{i+1}_inliers.png")
        )

        result = {
            "image_index": i + 1,
            "num_matches": len(good_matches),
            "num_inliers": num_inliers,
            "H": H
        }

        if H is not None:
            # Save matrix text
            save_homography_text(
                H,
                os.path.join(matrices_dir, f"H_ref_to_{i+1}.txt")
            )

            # Project boundary from reference to current view
            projected = cv2.perspectiveTransform(ref_corners, H)
            draw_projected_boundary(
                cur_img,
                projected,
                os.path.join(proj_dir, f"projected_boundary_on_image_{i+1}.png")
            )

            # Warp current image back to reference plane
            warped = warp_to_reference(cur_img, H, ref_img.shape)
            cv2.imwrite(os.path.join(warp_dir, f"warped_image_{i+1}.png"), warped)

            # Blend for approximate reconstruction
            blended = blend_images(blended, warped, alpha=0.35)
            cv2.imwrite(os.path.join(blend_dir, f"blend_up_to_image_{i+1}.png"), blended)

        homography_results.append(result)

    cv2.imwrite(os.path.join(args.output_dir, "final_blended_planar_reconstruction.png"), blended)

    # Camera info template for manual completion
    create_camera_info_template(
        image_paths,
        os.path.join(args.output_dir, "camera_info_template.json")
    )

    # Final summary report
    create_summary_report(
        os.path.join(args.output_dir, "part_b_report.txt"),
        image_paths,
        homography_results,
        ref_corners,
        note_manual_corners_used=(manual_corners is not None)
    )

    print("\nProcessing complete.")
    print(f"Outputs saved in: {args.output_dir}")
    print("\nImportant files:")
    print("- reference_boundary.png")
    print("- 03_matches/")
    print("- 04_projected_boundaries/")
    print("- 05_warped_to_reference/")
    print("- final_blended_planar_reconstruction.png")
    print("- part_b_report.txt")
    print("- camera_info_template.json")


if __name__ == "__main__":
    main()