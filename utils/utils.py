import os
import shutil
import zipfile


def unzip_file(file_path: str) -> None:
    """Unzips the given file at the same location

    Args:
        file_path (str): zip file location
    """
    if not file_path.endswith(".zip"):
        print("Not a zip file, aborting")

    # If not relative path expand it.
    if not os.path.isabs(file_path):
        file_path = os.path.expanduser(file_path)

    dir_name = file_path[:-4]
    os.mkdir(dir_name)

    with zipfile.ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(dir_name)


def get_kaggle_dataset(
    dataset: str, download_path: str = "~/Downloads/datasets/"
) -> None:
    """Downloads the given Kaggle Dataset and unzips it to folder

    Args:
        dataset (str): Kaggle Dataset url of type - "username/dataset"
        download_path (str, optional): Download directory. Defaults to "~/Downloads/datasets/".
    """
    _, dataset_name = dataset.split("/")

    # If not relative path expand it.
    if not os.path.isabs(download_path):
        download_path = os.path.expanduser("~/Downloads/datasets/")

    # if folder exists, halt
    if os.path.isdir(os.path.join(download_path, dataset_name)):
        raise ValueError("Dataset Folder already Exists.")
    else:
        print("Downloading dataset")
        return_value = os.popen(
            f"cd {download_path} && kaggle datasets download -d {dataset}"
        ).read()

        unzip_file(os.path.join(download_path, dataset_name + ".zip"))
        print("Download Complete")


def organize_face_mask_dataset(
    dataset_path: str, remove_augmented: bool = False
) -> None:
    """Organizes the face_mask dataset to the form accepted by keras.preprocessing.image_dataset_from_directory()

    Args:
        path (str): Dataset path
        remove_augmented (bool, optional): The dataset has pre-augmented images, pass True to remove them.
            Defaults to "~/Downloads/datasets/".
    """
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.expanduser(dataset_path)

    face_mask_ds_path = os.path.join(
        dataset_path, os.listdir(dataset_path)[0]
    )  # .../Face Mask Dataset/
    if os.path.basename(face_mask_ds_path) != "Face Mask Dataset":
        raise ValueError(
            "'Face Mask Dataset' directory not found, check if dataset_path is correct"
        )

    os.mkdir(os.path.join(dataset_path, "class0"))
    os.mkdir(os.path.join(dataset_path, "class1"))

    for path, _, files in os.walk(face_mask_ds_path, topdown=True):
        if files:
            # assign the class
            if os.path.basename(path) == "WithoutMask":
                class_name = "class0"
            elif os.path.basename(path) == "WithMask":
                class_name = "class1"
            else:
                raise ValueError("'WithoutMask' and 'WithMask' dirs not found.")

            # remove augmented
            if remove_augmented:
                files = [f for f in files if not f.startswith("Augmented")]

            # move them
            for file in files:
                file_path = os.path.join(path, file)
                move_to = os.path.join(dataset_path, class_name, file)
                # print(f"Moving file from: {file_path}\nto {move_to}\n")
                os.rename(file_path, move_to)

    shutil.rmtree(face_mask_ds_path, ignore_errors=True)
    print("Files Organized")


if __name__ == "__main__":
    pass
    # get_kaggle_dataset("ashishjangra27/face-mask-12k-images-dataset")
    # unzip_file("~/Downloads/datasets/face-mask-12k-images-dataset.zip")
    # organize_face_mask_dataset(
    #     "~/Downloads/datasets/face-mask-12k-images-dataset/")
