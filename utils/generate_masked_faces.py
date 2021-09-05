import glob
import cv2
import numpy as np
import mediapipe as mp
import os
import shutil


def process_mask(image_path: str, annotation_path: str):
    mask_img = cv2.imread(image_path, -1)
    mask_annotation = np.genfromtxt(annotation_path, delimiter=',', dtype=int)
    mask_annotation = np.delete(mask_annotation, [0, 3, 4, 5], axis=1)

    return mask_img, mask_annotation


def process_face(face_landmarks, f_height, f_width, lanmark_indices=[227, 195, 447, 58, 288, 152]):
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


def apply_mask(mask_img, mask_annotation, face_img, face_annotation):

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


def save(original_image_path, output_image, mask_idx, dataset_dir, move_original=False):
    """Saves the unmasked images to class0 folder and masked images to class1 folder. 
    This structure can be ready easily by'keras.preprocessing.image_dataset_from_directory()' function

    Args:
        original_image_path (str): Path to the original image; to move to class0
        output_image (cv2): The created masked iamge; to be pasted in class1 dir
        mask_idx (int): When multiple masks are applied, subscript the filename
        dataset_dir (str): The parent ds directory. Contains class0, class1 dirs
        move_original (bool, optional): If True, the original_image is moved instead of copied. Defaults to False.
    """
    filename = os.path.basename(original_image_path)

    # move(copy) original face_image from face_path to class0 dir
    if move_original:
        shutil.move(
            original_image_path,
            os.path.join(dataset_dir, "class0", filename))
    else:
        shutil.copyfile(
            original_image_path,
            os.path.join(dataset_dir, "class0", filename))

    # convert output_image into an image inside class1 dir
    cv2.imwrite(os.path.join(dataset_dir, "class1", filename), output_image)


def single_batch(face_paths, mask_paths, mask_annotation_paths, dataset_dir):
    # Get image and annotation of all the masks.
    mask_images = []
    mask_annotations = []
    for img_path, annotation_path in zip(mask_paths, mask_annotation_paths):
        img, ann = process_mask(img_path, annotation_path)
        mask_images.append(img)
        mask_annotations.append(ann)

    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
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
            for idx, mask_img, mask_ann in enumerate(zip(mask_images, mask_annotations)):
                output_img = face_img.copy()
                for face_ann in face_annotations:
                    # overwrite the output_img to apply masks to more than 1 face
                    output_img = apply_mask(
                        mask_img, mask_ann, output_img, face_ann)
                    save(f_path, output_img, dataset_dir, move_original=False)


def generate_masked_faces(face_ds_path, mask_ds_path, output_ds_path):
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
    elif not os.path.isdir(class1):
        os.mkdir(class1)

    # get list of all face_paths
    face_paths = []
    for i, path in enumerate(glob.iglob(face_ds_path + '/**/*.png', recursive=True)):
        if i < 10:
            face_paths.append(path)

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

    # single batch
    single_batch(face_paths, mask_paths, mask_annotation_paths, output_ds_path)


if __name__ == "__main__":
    generate_masked_faces(face_ds_path="~/Downloads/datasets/MaskedFaces-source/faces",
                          mask_ds_path="~/Downloads/datasets/MaskedFaces-source/masks",
                          output_ds_path="~/Downloads/datasets/MaskedFaces")
