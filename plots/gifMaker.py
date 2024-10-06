import numpy as np
import matplotlib.pyplot as plt
import imageio


def generate(data, title="", skip_frame=10):
    # Let's create a sample tensor for demonstration
    tensor = data
    f = data.shape[1]

    # List to store the file paths of the generated images
    filenames = []

    # Create each frame and save it as an image
    for i in range(tensor.shape[0]):
        if i % skip_frame != 0:
            continue
        if tensor[i].sum() == 0:
            continue
        fig_height = int(f ** (1 / 2))
        fig_width = int(f // fig_height)
        fig = plt.figure(figsize=(fig_width * 3, fig_height * 3))
        fig.suptitle(title + f"iteration: {i}", fontsize=20, fontweight="bold")

        j = 0
        for j in range(tensor.shape[1]):
            ax = fig.add_subplot(fig_height, fig_width, j + 1)
            ax.imshow(tensor[i][j], cmap="gray")
            ax.set_title(f"feature {j+1}")
            ax.axis("off")
            j += 1
            # Save the image to a temporary file
        filename = f"./temp/Frame {i+1}.png"
        filenames.append(filename)

        fig.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    # Create a GIF using the saved images
    gif_filename = "./tensor_animation_" + title.replace(" ", "-") + ".gif"
    with imageio.get_writer(gif_filename, mode="I", duration=0.005) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Optionally, you can clean up the individual frame files afterwards
    import os

    for filename in filenames:
        os.remove(filename)

    print(f"GIF created at {gif_filename}")
