import argparse
import os
import sys

# Constants for default file extensions
VISUALKERAS_SUFFIX = "_visualkeras.png"
STRUCTURE_SUFFIX = "_structure.png"


def main():
    parser = argparse.ArgumentParser(description="Visualize a Keras model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Keras model file (.h5 or .keras).",
    )
    args = parser.parse_args()

    # Deferred imports to avoid dependency issues if not installed
    try:
        import tensorflow as tf
        import visualkeras
    except ImportError as e:
        print(f"Error: Required libraries not found. {e}")
        print(
            "Please install them: pip install tensorflow visualkeras pillow pydot graphviz"
        )
        sys.exit(1)

    model_path = os.path.abspath(args.model_path)
    model_dir = os.path.dirname(model_path)
    base_name = os.path.basename(model_path)
    # Remove extension for image naming
    base_name_no_ext = os.path.splitext(base_name)[0]

    # Load Model
    try:
        print(f"Loading model from {model_path}...")
        # Check if file exists first
        if not os.path.exists(model_path):
            print(f"Error: File '{model_path}' does not exist.")
            sys.exit(1)

        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(
            f"Error: Could not load model. Ensure you are providing a full model file (.h5 or .keras), not just a weights file."
        )
        print(f"Specific error: {e}")
        sys.exit(1)

    # Visualization 1: 3D Layered View
    vk_output = os.path.join(model_dir, f"{base_name_no_ext}{VISUALKERAS_SUFFIX}")
    try:
        print(f"Generating 3D layered view: {vk_output}")
        visualkeras.layered_view(model, to_file=vk_output)
    except Exception as e:
        print(f"Warning: Failed to generate visualkeras 3D view. {e}")

    # Visualization 2: 2D Graph View
    struct_output = os.path.join(model_dir, f"{base_name_no_ext}{STRUCTURE_SUFFIX}")
    try:
        print(f"Generating 2D structure plot: {struct_output}")
        tf.keras.utils.plot_model(
            model,
            to_file=struct_output,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96,
        )
    except Exception as e:
        print(f"Warning: Failed to generate 2D structure plot. {e}")
        print("Ensure 'graphviz' and 'pydot' are installed and added to your PATH.")

    print("Done.")


if __name__ == "__main__":
    main()
