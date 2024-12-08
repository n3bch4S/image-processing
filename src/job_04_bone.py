import cv2
import numpy as np


def readim(image_path: str) -> np.ndarray:
    """
    Reads an X-ray image in grayscale.

    Args:
        image_path (str): Path to the X-ray image.

    Returns:
        numpy.ndarray: The grayscale image.

    Raises:
        FileNotFoundError: If the image file is not found.
    """
    # 1. Read the X-ray Film
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def prep(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    sharpening_factor: float = 0.5,
) -> np.ndarray:
    """
    Enhances the given X-ray image using CLAHE and applies sharpening.

    Args:
        image (numpy.ndarray): The grayscale image to enhance.
        clip_limit (float): Contrast limit for CLAHE. Default is 3.0.
        tile_grid_size (tuple): Size of the grid for CLAHE. Default is (8, 8).
        sharpening_factor (float): Factor by which sharpening is applied. Default is 1.5.

    Returns:
        numpy.ndarray: The enhanced and sharpened image.
    """
    # 2. Preprocessing - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)

    # 3. Apply sharpening to the enhanced image
    # Define a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Sharpen the image by applying the kernel
    sharpened_image = cv2.filter2D(enhanced_image, -1, sharpening_kernel)

    # Optionally, adjust sharpening strength by scaling the sharpened image
    # sharpened_image = cv2.addWeighted(
    #     enhanced_image, 1, sharpened_image, sharpening_factor, 0
    # )

    return sharpened_image


def filter_non_spine(
    enhanced_image: np.ndarray, threshold_value: int = 96
) -> tuple[np.ndarray, list]:
    """
    Filters out non-spine areas from the enhanced X-ray image by applying thresholding and contour detection.

    Args:
        enhanced_image (numpy.ndarray): The enhanced grayscale image.
        threshold_value (int): Threshold value for binary thresholding. Default is 128.

    Returns:
        numpy.ndarray: Binary mask of the spine region.
        list: Contours of the detected regions.
    """
    # Apply binary thresholding
    _, thresholded = cv2.threshold(
        enhanced_image, threshold_value, 255, cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a blank mask to draw the spine contour
    spine_mask = np.zeros_like(enhanced_image, dtype=np.uint8)

    # Filter contours based on size and location heuristics (e.g., aspect ratio or area)
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Filter small or irregular areas (you can adjust area threshold as needed)
        if area > 1000:  # Example area threshold
            cv2.drawContours(spine_mask, [contour], -1, color=255, thickness=-1)

    return spine_mask, contours


def spine_region(enhanced_image: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    """
    Extracts the spine region from the enhanced X-ray image using the largest contour.

    Args:
        enhanced_image (numpy.ndarray): The enhanced grayscale image.
        contours (list): List of contours detected in the image.

    Returns:
        numpy.ndarray: The isolated spine region as a grayscale image.
    """
    # Create a blank mask of the same size as the image
    spine_mask = np.zeros_like(enhanced_image, dtype=np.uint8)

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the mask
    cv2.drawContours(spine_mask, [largest_contour], -1, color=255, thickness=-1)

    # Apply the mask to the enhanced image to isolate the spine region
    spine_region = cv2.bitwise_and(enhanced_image, enhanced_image, mask=spine_mask)

    return spine_region


def segment(image: np.ndarray, thresholded: np.ndarray) -> np.ndarray:
    """
    Segments the spine bones using the watershed algorithm.

    Args:
        image (numpy.ndarray): The original image (used for watershed).
        thresholded (numpy.ndarray): Binary thresholded image.

    Returns:
        numpy.ndarray: The segmented image where each bone is assigned a unique label.
    """
    # Calculate the distance transform
    distance_transform = cv2.distanceTransform(thresholded, cv2.DIST_L2, 5)

    # Normalize the distance transform for visualization (optional)
    distance_transform_normalized = cv2.normalize(
        distance_transform, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Threshold the distance transform to find markers
    _, markers = cv2.connectedComponents(np.uint8(distance_transform_normalized))

    # Apply the watershed algorithm
    markers = markers + 1  # Increment marker values so background is not 0
    markers[thresholded == 0] = 0  # Ensure background pixels are set to 0
    segmented_bones = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)

    return segmented_bones


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
    """
    Displays multiple images in a single window with captions.

    Args:
        images (list of numpy.ndarray): List of images to display.
        captions (list of str): List of captions corresponding to the images.

    Raises:
        ValueError: If the number of images and captions do not match.
    """
    if len(images) != len(captions):
        raise ValueError("The number of images and captions must match.")

    # Resize all images to the same height for consistent display
    max_height = 300  # Desired display height for all images
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(max_height * aspect_ratio)
        resized_images.append(cv2.resize(img, (new_width, max_height)))

    # Add captions below images
    captioned_images = []
    for img, caption in zip(resized_images, captions):
        # Create a blank image for caption
        caption_height = 50  # Height for the caption area
        caption_img = np.zeros((caption_height, img.shape[1]), dtype=np.uint8)
        cv2.putText(
            caption_img, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2
        )

        # Stack the caption below the image
        captioned_image = np.vstack((img, caption_img))
        captioned_images.append(captioned_image)

    # Combine all images horizontally
    combined_image = np.hstack(captioned_images)

    # Display the combined image
    cv2.imshow("Images with Captions", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run():
    """
    Workflow runner function to read and preprocess an X-ray image.
    """
    try:
        # Call the readim() function
        image_name = "Image4"
        xray_image = readim(f"assets/project/{image_name}.jpeg")
        print("Image successfully read!")

        # Call the prep() function
        enhanced_image = prep(xray_image)
        print("Image successfully enhanced!")

        # Call the filter_non_spine() function
        spine_mask, contours = filter_non_spine(enhanced_image)
        print("Non-spine areas filtered!")

        # Call the spine_region() function
        isolated_spine = spine_region(enhanced_image, contours)
        print("Spine region successfully isolated!")

        # Call the segment() function
        segmented_bones = segment(enhanced_image, spine_mask)
        print("Spine bones successfully segmented!")

        # Call the save_vertebra() function to save the segmented vertebrae
        save_vertebra(segmented_bones, image_name)

        # Display images using show_images()
        show_images(
            [
                xray_image,
                enhanced_image,
                spine_mask,
                isolated_spine,
                segmented_bones.astype(np.uint8) * 50,
            ],
            [
                "Original Image",
                "Enhanced Image",
                "Spine Mask",
                "Isolated Spine",
                "Segmented Bones",
            ],
        )

    except FileNotFoundError as e:
        print(e)
