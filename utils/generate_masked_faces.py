import glob
from typing import Tuple
import cv2
import numpy as np
import mediapipe as mp
import os
import concurrent.futures


def process_mask(image_path: str, annotation_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get the mask_image and its corresponding annotation, mask_annotation

    Args:
        image_path (str): Abosolute path of mask's image
        annotation_path (str): Absolute path of mask's annotation .csv file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Return the mask image and the matrix of annotation
    """
    mask_img = cv2.imread(image_path, -1)
    mask_annotation = np.genfromtxt(annotation_path, delimiter=',', dtype=int)
    mask_annotation = np.delete(mask_annotation, [0, 3, 4, 5], axis=1)

    return mask_img, mask_annotation


def process_face(face_landmarks, f_height: int, f_width: int, lanmark_indices: list = [227, 195, 447, 58, 288, 152]) -> list:
    """Get the list of annotations(only for the indices given in landmark_indices) for all the faces for a single image (the face's landmarks are passed).

    Args:
        face_landmarks (mediapipe): The object containing the 468 landmarks for all the faces in a single image
        f_height (int): Face image's height
        f_width (int): Face image's width
        lanmark_indices (list, optional): The handpicked index of annotations that corresponds to the annotations made on the mask. Defaults to [227, 195, 447, 58, 288, 152].

    Returns:
        list: Each element in the list corresponds to the annotaions for a single face.
    """
    # list of np array containing the annotations; 1 annotation array for each face
    annotation_list = []

    for face in face_landmarks.multi_face_landmarks:
        # Get annotation for each face in the image
        ann = []
        landmarks = [face.landmark[i] for i in lanmark_indices]

        #  get absolute (x,y) for each landmark
        ann = [
            np.array([int(landmark.x * f_width), int(landmark.y * f_height)])
            for landmark in landmarks
        ]
        annotation_list.append(np.array(ann))

    return annotation_list


def apply_mask(mask_img: np.ndarray, mask_annotation: np.ndarray, face_img: np.ndarray, face_annotation: np.ndarray) -> np.ndarray:
    """Warp the mask_img using the mask_annotation and face_annotation and apply it to the face_img. Return the final combined image.

    Args:
        mask_img (np.ndarray): Image of the mask.
        mask_annotation (np.ndarray): Array of (x,y) coordinates of the annotations made on the mask.
        face_img (np.ndarray): Image of the face.
        face_annotation (np.ndarray): Array of (x,y) coordinates of the annotations on the face.

    Returns:
        np.ndarray: The final image where the mask is applied to the face.
    """

    f_height, f_width = face_img.shape[:2]
    hom = cv2.findHomography(mask_annotation, face_annotation)[0]
    warped = cv2.warpPerspective(
        mask_img, hom, (f_width, f_height), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # Get alpha channel (index 3)
    alpha_channel = warped[:, :, -1]
    # Copy and convert the mask to a float and give it 3 channels
    alpha_channel_scaled = alpha_channel.copy() / 255.0
    alpha_channel_scaled = np.dstack([alpha_channel_scaled] * 3)
    # Remove the alpha channel from the warped image
    warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)

    # Make the bg of mask tranparent
    warped_multiplied = cv2.multiply(
        alpha_channel_scaled, warped.astype("float"))
    # Make the location of the mask on face transparent
    image_multiplied = cv2.multiply(
        face_img.astype(float), 1.0 - alpha_channel_scaled)
    # Add to get the final image
    final_image = cv2.add(warped_multiplied, image_multiplied).astype("uint8")

    return final_image


def save(face_image: np.ndarray, face_image_filename: str, output_image: np.ndarray, mask_idx: int, output_resolution_fr: float, dataset_dir: str) -> None:
    """Saves the unmasked images to class0 folder and masked images to class1 folder. 
    This structure can be ready easily by'keras.preprocessing.image_dataset_from_directory()' function

    Args:
        face_image (np.ndarray): The original face image.
        face_imgae_filename (str): Filename of face image.
        output_image (np.ndarray): The created masked image; to be pasted in class1 dir
        mask_idx (int): When multiple masks are applied, subscript the filename
        output_resolution_fr (float): Resize the final image to output_resolution_fr*original dimensions; e.g., 0.5*(1024x1024)=512x512
        dataset_dir (str): The parent ds directory. Contains class0, class1 dirs
    """
    if mask_idx >= 1:
        parts = os.path.splitext(face_image_filename)
        face_image_filename = parts[0] + f"_{mask_idx}" + parts[1]

    # Write the original image only once(since this is called more than once if more masks are present)
    if mask_idx == 0:
        face_image = cv2.resize(face_image, (0, 0),
                                fx=output_resolution_fr, fy=output_resolution_fr)
        cv2.imwrite(os.path.join(dataset_dir, "class0", face_image_filename),
                    face_image)

    # convert output_image into an image inside class1 dir
    output_image = cv2.resize(output_image, (0, 0),
                              fx=output_resolution_fr, fy=output_resolution_fr)
    cv2.imwrite(os.path.join(dataset_dir, "class1", face_image_filename),
                output_image)


def single_batch(face_paths: list, mask_paths: list, mask_annotation_paths: list, dataset_dir: str, output_resolution_fr: float) -> None:
    """Given a list of paths to the face image, the mask image and mask_annotation. It applies all the masks to all the images
    and saves them in the dataset_dir

    Args:
        face_paths (list): List of Paths to face images.
        mask_paths (list): List of Paths to mask images.
        mask_annotation_paths (list): List of Paths to mask annotations.
        dataset_dir (str): Directory to which the outputs are stored.
        output_resolution_fr (float): Resize the final image to output_resolution_fr*original dimensions; e.g., 0.5*(1024x1024)=512x512
    """
    # Get image and annotation of all the masks.
    mask_images = []
    mask_annotations = []
    for img_path, annotation_path in zip(mask_paths, mask_annotation_paths):
        img, ann = process_mask(img_path, annotation_path)
        mask_images.append(img)
        mask_annotations.append(ann)

    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as mesh_model:

        # Apply mask for all images.
        for f_path in face_paths:
            print(f"Processing file: {os.path.basename(f_path)}")
            face_img = cv2.imread(f_path)
            f_height, f_width = face_img.shape[:2]

            face_landmarks = mesh_model.process(face_img)
            if not face_landmarks.multi_face_landmarks:
                continue

            face_annotations = process_face(face_landmarks, f_height, f_width)
            # Apply all the types of masks for each face
            for idx, (mask_img, mask_ann) in enumerate(zip(mask_images, mask_annotations)):
                output_img = face_img.copy()
                for face_ann in face_annotations:
                    # overwrite the output_img to apply masks to more than 1 face
                    output_img = apply_mask(
                        mask_img, mask_ann, output_img, face_ann)
                    save(
                        face_image=face_img,
                        face_image_filename=os.path.basename(f_path),
                        output_image=output_img,
                        mask_idx=idx,
                        output_resolution_fr=output_resolution_fr,
                        dataset_dir=dataset_dir)


def generate_masked_faces(face_ds_path: str, mask_ds_path: str, output_ds_path: str, output_resolution_fr: float = 0.5, n_images: int = 1000, n_process: int = 8) -> None:
    """The main function which processes the entire dataset.

    Args:
        face_ds_path (str): Path to Face images. Can have multiple sub dirs.
        mask_ds_path (str): Path to Mask images; no subdirs; images->.png annotation->.csv; same name for image and annotations.
        output_ds_path (str): Path to where the final dataset is created.
        output_resolution_fr (float): Resize the final image to output_resolution_fr*original dimensions; e.g., 0.5*(1024x1024)=512x512
        n_images (int): The number of face images to process.
        n_process (int): Number of processos to spin up. Multiprocessing.
    """
    # convert to absolute paths
    if not os.path.isabs(output_ds_path):
        output_ds_path = os.path.expanduser(output_ds_path)
    if not os.path.isabs(face_ds_path):
        face_ds_path = os.path.expanduser(face_ds_path)
    if not os.path.isabs(mask_ds_path):
        mask_ds_path = os.path.expanduser(mask_ds_path)

    # create class directories if it does not exist
    class0 = os.path.join(output_ds_path, "class0")
    class1 = os.path.join(output_ds_path, "class1")
    if not os.path.isdir(class0):
        os.mkdir(class0)
    if not os.path.isdir(class1):
        os.mkdir(class1)

    if n_images < n_process:
        raise ValueError("The n_images must be greater than n_process")

    # face_paths: list[list]; each element is a list of images to be assinged for 1 process
    # the image paths are put in a bucket, which is filled until it reaches the images_per_process
    # then it is appended to face_paths
    face_paths = []
    images_per_process = int(n_images/n_process)
    bucket = []
    bucket_counter = 0
    item_per_bucket = 0
    for i, path in enumerate(glob.iglob(face_ds_path + '/**/*.png', recursive=True)):
        if i == n_images:
            # after reaching the end, dump the last bucket since it was kept getting filled.
            if item_per_bucket > 0:
                face_paths.append(bucket)
            break
        # each bucket can be filled up with the images_per_process items.
        # if it reaches that threshold, append it to the main list and empty it.
        if item_per_bucket >= images_per_process:
            # but if its the last bucket, fill it up until we run out of images.
            # this is the case when n_images % n_process != 0
            if bucket_counter == n_process-1:
                bucket.append(path)
                item_per_bucket += 1
                continue
            # dump the bucket
            face_paths.append(bucket)
            bucket = []
            item_per_bucket = 0
            bucket_counter += 1

        # add to bucket
        bucket.append(path)
        item_per_bucket += 1

    # get list of all mask_paths, mask_annotation_paths
    mask_paths = []
    mask_annotation_paths = []
    for path in glob.glob(mask_ds_path + "/*"):
        if path.endswith(".png"):
            mask_paths.append(path)
        elif path.endswith(".csv"):
            mask_annotation_paths.append(path)
        else:
            print("Unknown file type in mask_ds_path")
    mask_paths.sort()
    mask_annotation_paths.sort()
    assert len(mask_paths) == len(mask_annotation_paths)

    # Spawn multiple processes.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures_objs = []
        for i in range(n_process):
            futures_objs.append(executor.submit(single_batch, face_paths[i], mask_paths, mask_annotation_paths,
                                                output_ds_path, output_resolution_fr))


if __name__ == "__main__":
    generate_masked_faces(face_ds_path="~/Downloads/datasets/MaskedFaces-source/faces",
                          mask_ds_path="~/Downloads/datasets/MaskedFaces-source/masks",
                          output_ds_path="~/Downloads/datasets/MaskedFaces",
                          output_resolution_fr=0.3,
                          n_images=5000,
                          n_process=10)
