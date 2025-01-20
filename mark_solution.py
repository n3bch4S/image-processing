import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(image_count, prefix="Image", extension="jpeg"):
    return [
        cv2.imread(f"{prefix} {i + 1}.{extension}", cv2.IMREAD_GRAYSCALE)
        for i in range(image_count)
    ]


def crop_images(images, crop_ratio=0.25):
    cropped_images = []
    for image in images:
        h, w = image.shape
        left_crop = int(crop_ratio * w)
        right_crop = int((1 - crop_ratio) * w)
        cropped_images.append(image[:, left_crop:right_crop])
    return cropped_images


def display_images(images, titles, rows, cols, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def enhance_image_intensity(images):
    enhanced_images = []
    avg_intensities = []
    for image in images:
        avg_intensity = int(np.mean(image))
        avg_intensities.append(avg_intensity)
        threshold = int(0.2 * image.shape[0])
        parts = [image[i * threshold : (i + 1) * threshold, :] for i in range(5)]
        modified_parts = [
            cv2.equalizeHist(
                np.where(part < int(np.mean(part)), int(np.mean(part)), part)
            )
            for part in parts
        ]
        enhanced_images.append(np.vstack(modified_parts))
    return enhanced_images, avg_intensities


def save_subregions_in_image(image, box_pos, image_name):
    for i, (x, y, w, h) in enumerate(box_pos):
        try:
            sub_region = image[y - 5 : y + h + 5, x - 5 : x + w + 5]
        except:
            sub_region = image[y : y + h, x : x + w]
        sub_region_filename = f"{image_name}_{i+1}.jpg"
        # cv2.imwrite(sub_region_filename, sub_region)


def draw_bounding_boxes(images, intensities, threshold_offset=80):
    images_with_boxes = []
    binary_images = []
    box_poses = []
    image_names = ["image 1", "image 2", "image 3", "image 4"]
    for image, image_name, intensity in zip(images, image_names, intensities):
        _, binary_image = cv2.threshold(
            image, intensity + threshold_offset, 255, cv2.THRESH_BINARY
        )

        # kernel = np.ones((3, 3), np.uint8)
        # closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # TODO: version 2 use canny
        blurred_image = cv2.GaussianBlur(binary_image, (11, 11), 0)
        edges = cv2.Canny(blurred_image, 150, 250)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # TODO: version 1 use bin threshold
        binary_images.append(binary_image)
        # contours, _ = cv2.findContours(
        #     binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        box_pos = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 10 < w < 60 and 10 < h < 60 and 1.1 < w / h < 3.0 and x > 10:
                cv2.rectangle(
                    image_with_boxes,
                    (x - 5, y - 5),
                    (x + w + 10, y + h + 10),
                    (0, 255, 0),
                    2,
                )
                box_pos.append((x, y, w, h))
        save_subregions_in_image(image, box_pos, image_name)
        images_with_boxes.append(image_with_boxes)
        box_poses.append(box_pos)
    return images_with_boxes, binary_images, box_poses


def main():
    image_list = load_images(4)
    cropped_images = crop_images(image_list)
    display_images(
        image_list + cropped_images,
        [f"Original Image {i + 1}" for i in range(4)]
        + [f"Cropped Image {i + 1}" for i in range(4)],
        2,
        4,
    )

    enhanced_images, avg_intensities = enhance_image_intensity(cropped_images)
    display_images(
        enhanced_images, [f"Modified Image {i + 1} with AVG+20" for i in range(4)], 1, 4
    )

    images_with_boxes, binary_images, box_poses = draw_bounding_boxes(
        enhanced_images, avg_intensities
    )
    display_images(
        binary_images + images_with_boxes,
        [f"Thresh Image {i + 1}" for i in range(4)]
        + [f"Image {i + 1} with Bounding Boxes" for i in range(4)],
        2,
        4,
        figsize=(16, 8),
    )


if __name__ == "__main__":
    main()
