import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def img_region_grow(img, seeds):
    rows, cols = img.shape
    tempmat = np.zeros((rows, cols), dtype=np.int8)

    seeds_np = np.array(seeds).T
    if seeds_np.shape[0] != 2:
        raise ValueError("Invalid dimension of 'seeds'")

    sc = seeds_np.shape[1]
    vals = np.zeros(sc)

    for i in range(sc):
        x = seeds_np[0, i]
        y = seeds_np[1, i]
        vals[i] = img[x, y]
        tempmat[x, y] = 1

    mu = np.mean(vals)
    sigma = np.std(vals)
    Z = 2
    dv = max(50, Z * sigma)
    thmin = max(0, mu - dv)
    thmax = min(255, mu + dv)

    newp = seeds_np.copy()
    modified = True

    while modified:
        modified = False
        pnewp = newp.copy()
        newp = np.empty((2, 0), dtype=int)

        for i in range(pnewp.shape[1]):
            x = pnewp[0, i]
            y = pnewp[1, i]

            for xn in range(max(0, x - 1), min(rows, x + 2)):
                for yn in range(max(0, y - 1), min(cols, y + 2)):
                    if tempmat[xn, yn] != 0:
                        continue

                    PIX = img[xn, yn]
                    if thmin <= PIX <= thmax:
                        tempmat[xn, yn] = 1
                        newp = np.hstack((newp, np.array([[xn], [yn]])))
                        modified = True
                    else:
                        tempmat[xn, yn] = -1

    bwimg = tempmat == 1
    return bwimg


def load_input_image(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
    return None


def process_single_image(image):
    h, w = image.shape
    # Cross pattern seeds
    seeds = [
        (h // 4, w // 2),
        (h // 2, w // 4),
        (h // 2, 3 * w // 4),
        (3 * h // 4, w // 2),
    ]

    segmented = img_region_grow(image, seeds)
    save_essential_outputs(image, segmented, seeds)
    return segmented


def save_essential_outputs(image, segmented, seeds):
    # Save original grayscale image (without seed points)
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap="gray")
    plt.title("Original Grayscale Image")
    plt.axis("off")
    plt.savefig("outputs/q2_original.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save seed points visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap="gray")
    plt.scatter(
        [s[1] for s in seeds],
        [s[0] for s in seeds],
        c="red",
        s=60,
        marker="x",
        linewidths=3,
    )
    for i, seed in enumerate(seeds):
        plt.text(
            seed[1] + 5,
            seed[0] + 5,
            f"S{i+1}",
            color="red",
            fontweight="bold",
            fontsize=10,
        )
    plt.title("Seed Points for Cross Pattern Region Growing")
    plt.axis("off")
    plt.savefig("outputs/q2_seeds.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save cross pattern segmentation result
    plt.figure(figsize=(8, 6))
    plt.imshow(segmented, cmap="gray")
    plt.title("Cross Pattern Region Growing Segmentation")
    plt.axis("off")
    plt.savefig("outputs/q2_segmented.png", dpi=300, bbox_inches="tight")
    plt.close()


def demonstrate_region_growing():
    input_image_path = "inputs/input_image_q2.jpg"
    image = load_input_image(input_image_path)

    if image is not None:
        if image.shape[0] > 400 or image.shape[1] > 400:
            scale = min(400 / image.shape[0], 400 / image.shape[1])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
    else:
        return

    process_single_image(image)


def main():
    demonstrate_region_growing()


if __name__ == "__main__":
    main()
