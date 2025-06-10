# Align Mesh to Origin Coordinates

## Overview

This script allows you to load your mesh and align it with the origin coordinates. It's a handy tool for those who work in reverse engineering, or any field that requires precise mesh alignment.

## Prerequisites

To use this script, you must have Python installed on your computer along with the following libraries:

```shell
pip install numpy pyvista scipy scikit-learn
```

## Usage Instructions

1. Select your mesh file as a .ply file. This will be the input mesh you wish to align.

2. Choose at least three points on the mesh to define a plane. You can add as many points as you like for more accurate alignment.

3. Confirm the alignment of the selected plane to one of the world coordinate system axes by pressing `x`, `y`, or `z`.

4. Select points again for the second axis. Make sure to confirm the selection with `x`, `y`, or `z`, ensuring that this axis is different from the first one chosen.

5. Once the second axis is selected and confirmed, the mesh will be aligned according to these two axes.

6. Press the `s` key to save the aligned mesh. It will be saved as `transformed_mesh.stl` in the same folder.
7. Use the `r` key to reset the current selection or `b` to remove the last picked point.

## Screenshot

Here's a screenshot to help you visualize the process:

![Preview](preview.gif)
