import fiftyone as fo
import fiftyone.zoo as foz
import cv2
import numpy as np

import fiftyone.brain as fob
import pandas as pd


def test():
    dataset = foz.load_zoo_dataset("quickstart")
    session = fo.launch_app(dataset)

    # Image embeddings
    fob.compute_visualization(dataset, brain_key="img_viz")

    # Object patch embeddings
    # compute other embeddings
    model = foz.load_zoo_model("dinov2-vitg14-torch")
    embeddings_dinov2 = dataset.compute_embeddings(model)
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings_dinov2,
        num_dims=2,
        method="umap",
        brain_key="dinov2_test",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("dinov2_test")

    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings_dinov2,
        num_dims=4,
        method="umap",
        brain_key="dinov2_test_4dim",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("dinov2_test_4dim")

    session = fo.launch_app(dataset)

    embeddings_panel = fo.Panel(
        type="Embeddings",
        state=dict(brainResult="dinov2_test", colorByField="uniqueness"),
    )


def test_mnist():
    dataset = foz.load_zoo_dataset("mnist")
    test_split = dataset.match_tags("test")

    # Construct a ``num_samples x num_pixels`` array of images
    embeddings = np.array([
        cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel()
        for f in test_split.values("filepath")
    ])

    # compute other embeddings
    model = foz.load_zoo_model("dinov2-vitg14-torch")
    embeddings_dinov2 = test_split.compute_embeddings(model)

    # compute 2D representation
    results = fob.compute_visualization(
        test_split,
        embeddings=embeddings,
        num_dims=2,
        method="umap",
        brain_key="mnist_test",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("mnist_test")

    results_dinov2 = fob.compute_visualization(
        test_split,
        embeddings=embeddings_dinov2,
        num_dims=2,
        method="umap",
        brain_key="mnist_test_dinov2",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("mnist_test")
    dataset.load_brain_results("mnist_test_dinov2")

    # visualising embeddings
    # Launch App instance
    session = fo.launch_app(test_split)
    embeddings_panel = fo.Panel(
        type="Embeddings",
        state=dict(brainResult="img_viz", colorByField="uniqueness"),
    )


def test_rice():
    rice_path = "data/Rice.v2i.multiclass/test"
    class_csv_path = "data/Rice.v2i.multiclass/test/_classes.csv"

    dataset = fo.Dataset.from_images_dir(rice_path)
    class_df = pd.read_csv(class_csv_path)

    category_list = []
    for sample in dataset:
        sample_path = sample.filepath.split("/")[-1]
        sample_info = class_df.query(f"filename == '{sample_path}'").iloc[0].iloc[1:]
        categories = sample_info.index[sample_info == 1].tolist()
        sample.tags= categories
        sample.save()

    # Construct a ``num_samples x num_pixels`` array of images
    # compute other embeddings
    model = foz.load_zoo_model("dinov2-vitg14-torch")
    embeddings_dinov2 = dataset.compute_embeddings(model)

    results_dinov2 = fob.compute_visualization(
        dataset,
        embeddings=embeddings_dinov2,
        num_dims=2,
        method="umap",
        brain_key="test_dinov2",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("test_dinov2")

    # add other embeddings
    model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    embeddings_mobilenet = dataset.compute_embeddings(model)

    results_mobilenet = fob.compute_visualization(
        dataset,
        embeddings=embeddings_mobilenet,
        num_dims=2,
        method="umap",
        brain_key="test_mobilenet",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("test_mobilenet")

    # add other embeddings
    model = foz.load_zoo_model('resnet34-imagenet-torch')
    embeddings_mobilenet = dataset.compute_embeddings(model)

    results_mobilenet = fob.compute_visualization(
        dataset,
        embeddings=embeddings_mobilenet,
        num_dims=2,
        method="umap",
        brain_key="test_resnet34",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("test_resnet34")

    # visualising embeddings
    # Launch App instance

    embeddings_panel = fo.Panel(
        type="Embeddings",
        state=dict(brainResult="test_resnet34", colorByField="uniqueness"),
    )
    session = fo.launch_app(dataset)


def test_rice2():
    rice_path = "data/Rice.v2i.multiclass/test"
    rice_path_val = "data/Rice.v2i.multiclass/valid"
    class_csv_path = "data/Rice.v2i.multiclass/test/_classes.csv"
    class_csv_path_val = "data/Rice.v2i.multiclass/valid/_classes.csv"

    dataset_test = fo.Dataset.from_images_dir(rice_path)
    dataset_val = fo.Dataset.from_images_dir(rice_path_val)
    class_df = pd.read_csv(class_csv_path)
    class_df_val = pd.read_csv(class_csv_path_val)

    category_list = []
    for sample in dataset_test:
        sample_path = sample.filepath.split("/")[-1]
        sample_info = class_df.query(f"filename == '{sample_path}'").iloc[0].iloc[1:]
        categories = sample_info.index[sample_info == 1].tolist()
        sample.tags = categories
        sample.save()

    category_list = []
    for sample in dataset_val:
        sample_path = sample.filepath.split("/")[-1]
        sample_info = class_df_val.query(f"filename == '{sample_path}'").iloc[0].iloc[1:]
        categories = sample_info.index[sample_info == 1].tolist()
        sample.tags = categories
        sample.save()

    dataset_val.merge_samples(dataset_test)
    dataset = dataset_val

    # Construct a ``num_samples x num_pixels`` array of images
    # compute other embeddings
    model = foz.load_zoo_model("dinov2-vitg14-torch")
    embeddings_dinov2 = dataset.compute_embeddings(model)

    results_dinov2 = fob.compute_visualization(
        dataset,
        embeddings=embeddings_dinov2,
        num_dims=2,
        method="umap",
        brain_key="test_dinov2",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("test_dinov2")

    # add other embeddings
    model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    embeddings_mobilenet = dataset.compute_embeddings(model)

    results_mobilenet = fob.compute_visualization(
        dataset,
        embeddings=embeddings_mobilenet,
        num_dims=2,
        method="umap",
        brain_key="test_mobilenet",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("test_mobilenet")

    # add other embeddings
    model = foz.load_zoo_model('resnet34-imagenet-torch')
    embeddings_mobilenet = dataset.compute_embeddings(model)

    results_mobilenet = fob.compute_visualization(
        dataset,
        embeddings=embeddings_mobilenet,
        num_dims=2,
        method="umap",
        brain_key="test_resnet34",
        verbose=True,
        seed=51
    )
    dataset.load_brain_results("test_resnet34")

    # visualising embeddings
    # Launch App instance

    embeddings_panel = fo.Panel(
        type="Embeddings",
        state=dict(brainResult="test_resnet34", colorByField="uniqueness"),
    )
    session = fo.launch_app(dataset)


def main():
    test()