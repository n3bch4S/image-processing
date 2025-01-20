import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_ubyte


def open_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def enhance_image(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)

    sharpening_kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ]
    )
    enhanced_image = cv2.filter2D(enhanced_image, -1, sharpening_kernel)

    scharr_x = cv2.Scharr(enhanced_image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(enhanced_image, cv2.CV_64F, 0, 1)
    scharr_combined = cv2.magnitude(scharr_x, scharr_y)
    enhanced_image = cv2.normalize(scharr_combined, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced_image


# def filter_non_spine(
#     enhanced_image: np.ndarray,
#     method: str = "threshold",
#     threshold_value: int = 150,
#     canny_thresholds: tuple[int, int] = (40, 100),
# ) -> tuple[np.ndarray, list, np.ndarray]:
#     """
#     Filters out non-spine areas from the enhanced X-ray image using a specified method and finds contours.

#     Args:
#         enhanced_image (numpy.ndarray): The enhanced grayscale image.
#         method (str): Filtering method to use ("threshold", "canny", etc.). Default is "canny".
#         threshold_value (int): Threshold value for binary thresholding. Used if method is "threshold". Default is 127.
#         canny_thresholds (tuple[int, int]): Lower and upper thresholds for Canny edge detection. Default is (50, 150).

#     Returns:
#         numpy.ndarray: Binary mask of the spine region.
#         list: Contours of the detected regions.
#         numpy.ndarray: Binary image after applying the chosen filtering method.
#     """
#     if method == "threshold":
#         # Apply binary thresholding
#         _, binary_image = cv2.threshold(
#             enhanced_image, threshold_value, 255, cv2.THRESH_BINARY
#         )
#     elif method == "canny":
#         # Apply Canny edge detection
#         binary_image = cv2.Canny(
#             enhanced_image, canny_thresholds[0], canny_thresholds[1]
#         )
#     else:
#         raise ValueError(
#             f"Unknown method: {method}. Supported methods are 'threshold' and 'canny'."
#         )

#     # Find contours
#     contours, _ = cv2.findContours(
#         binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     # Create a blank mask to draw the spine contour
#     spine_mask = np.zeros_like(enhanced_image, dtype=np.uint8)

#     # Filter contours based on size and location heuristics (e.g., aspect ratio or area)
#     for contour in contours:
#         # Calculate contour area
#         area = cv2.contourArea(contour)

#         # Filter small or irregular areas (adjust area threshold as needed)
#         if area > 1000:  # Example area threshold
#             cv2.drawContours(spine_mask, [contour], -1, color=255, thickness=-1)

#     return spine_mask, contours, binary_image


# def spine_region(enhanced_image: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
#     """
#     Extracts the spine region from the enhanced X-ray image using the largest contour.

#     Args:
#         enhanced_image (numpy.ndarray): The enhanced grayscale image.
#         contours (list): List of contours detected in the image.

#     Returns:
#         numpy.ndarray: The isolated spine region as a grayscale image.
#     """
#     # Create a blank mask of the same size as the image
#     spine_mask = np.zeros_like(enhanced_image, dtype=np.uint8)

#     # Find the largest contour by area
#     largest_contour = max(contours, key=cv2.contourArea)

#     # Draw the largest contour on the mask
#     cv2.drawContours(spine_mask, [largest_contour], -1, color=255, thickness=-1)

#     # Apply the mask to the enhanced image to isolate the spine region
#     spine_region = cv2.bitwise_and(enhanced_image, enhanced_image, mask=spine_mask)

#     return spine_region


# def segment(
#     image: np.ndarray,
#     method: str = "superpixel",
#     k_clusters: int = 3,
#     seed_point: tuple = (100, 100),
#     slic_segments: int = 100,
#     slic_compactness: float = 1.2,
# ) -> np.ndarray:
#     """
#     Segments the spine bones using the specified segmentation algorithm.

#     Args:
#         image (numpy.ndarray): The original image (used for segmentation).
#         method (str): Segmentation method to use ("watershed", "kmeans", "region_growing", "superpixel").
#         k_clusters (int): Number of clusters for K-Means segmentation. Used only if method is "kmeans".
#         seed_point (tuple): Seed point for region growing. Used only if method is "region_growing".
#         slic_segments (int): Number of segments for SLIC superpixel. Used only if method is "superpixel".
#         slic_compactness (float): Compactness parameter for SLIC superpixel. Used only if method is "superpixel".

#     Returns:
#         numpy.ndarray: The segmented image where each bone is assigned a unique label.
#     """
#     if method == "watershed":
#         # Convert image to binary using Otsu's method
#         _, binary_image = cv2.threshold(
#             image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#         )

#         # Calculate the distance transform
#         distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

#         # Normalize the distance transform for marker detection
#         distance_transform_normalized = cv2.normalize(
#             distance_transform, None, 0, 255, cv2.NORM_MINMAX
#         ).astype(np.uint8)

#         # Threshold the distance transform to find markers
#         _, markers = cv2.connectedComponents(np.uint8(distance_transform_normalized))

#         # Apply the watershed algorithm
#         markers = markers + 1  # Increment marker values so background is not 0
#         markers[binary_image == 0] = 0  # Ensure background pixels are set to 0
#         segmented_bones = cv2.watershed(
#             cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers
#         )

#     elif method == "kmeans":
#         # Reshape the image for K-Means clustering
#         reshaped_image = image.reshape((-1, 1)).astype(np.float32)

#         # Define criteria and apply K-Means
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#         _, labels, _ = cv2.kmeans(
#             reshaped_image,
#             k_clusters,
#             None,
#             criteria,
#             10,
#             cv2.KMEANS_RANDOM_CENTERS,
#         )

#         # Reshape labels back to the original image shape
#         segmented_bones = labels.reshape(image.shape)

#     elif method == "region_growing":
#         # Use Region Growing based on a seed point
#         seed_value = image[seed_point]
#         mask = np.zeros_like(image, dtype=np.uint8)

#         # Parameters for region growing
#         lower_bound = seed_value - 20
#         upper_bound = seed_value + 20
#         connectivity = 4

#         # Apply region growing
#         flood_flags = connectivity | cv2.FLOODFILL_FIXED_RANGE
#         _, mask, _, _ = cv2.floodFill(
#             image.copy(),
#             mask,
#             seedPoint=seed_point,
#             newVal=255,
#             loDiff=lower_bound,
#             upDiff=upper_bound,
#             flags=flood_flags,
#         )
#         segmented_bones = mask

#     elif method == "superpixel":
#         # Convert grayscale image to RGB for SLIC (Superpixel Segmentation)
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         segments = slic(
#             rgb_image,
#             n_segments=slic_segments,
#             compactness=slic_compactness,
#             start_label=1,
#         )
#         segmented_bones = img_as_ubyte(segments)

#     else:
#         raise ValueError(
#             f"Unknown method: {method}. Supported methods are 'watershed', 'kmeans', 'region_growing', and 'superpixel'."
#         )

#     return segmented_bones


def save_vertebra(segmented_bones: np.ndarray, image_name: str) -> None:
    """
    Saves each segmented vertebra as an individual image with a label based on its region ID.

    Args:
        segmented_bones (numpy.ndarray): The result of the watershed segmentation containing labeled regions.

    Returns:
        None
    """
    # Define a dictionary mapping region IDs to vertebrae labels
    labels = {
        1: "C1",
        2: "T2",
        3: "L3",
        4: "L4",
        5: "L5",
        6: "S1",  # Example mapping, can expand as needed
    }

    # Iterate through each label and region ID
    for region_id, label in labels.items():
        # Create a mask for each region by comparing with the region ID
        region_mask = (segmented_bones == region_id).astype(np.uint8) * 255

        # Check if the region is present in the segmentation (non-empty mask)
        if np.sum(region_mask) > 0:
            # Save the region mask as a JPEG file
            cv2.imwrite(f"result/project/{image_name}/{label}.jpg", region_mask)
            print(f"Saved result/project/{image_name}/{label}.jpg")
        else:
            print(f"Region {label} not found in the segmentation.")


def show_images(images, captions):
    if len(images) != len(captions):
        raise ValueError("The number of images and captions must match.")

    plt.figure(figsize=(16, 8))

    for image, caption in zip(images, captions):
        0
    plt.show()


def main():
    prefix_name = "Image"
    xray_image = open_image(f"assets/project/{image_name}.jpeg")

    enhanced_image = enhance_image(xray_image)

    # # Call the filter_non_spine() function
    # spine_mask, contours, binary_image = filter_non_spine(enhanced_image)
    # print("Non-spine areas filtered!")

    # # Call the spine_region() function
    # isolated_spine = spine_region(enhanced_image, contours)
    # print("Spine region successfully isolated!")

    # # Call the segment() function
    # segmented_bones = segment(enhanced_image)
    # print("Spine bones successfully segmented!")

    # Call the save_vertebra() function to save the segmented vertebrae
    # save_vertebra(segmented_bones, image_name)

    # Display images using show_images()
    show_images(
        [
            xray_image,
            enhanced_image,
        ],
        [
            "Original Image",
            "Enhanced Image",
        ],
    )


if __name__ == "__main__":
    main()
