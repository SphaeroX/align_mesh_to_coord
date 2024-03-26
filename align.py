# Notwendige Bibliotheken
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog

# Globale Variablen
picked_points = []
mesh_transformed = None
plane_points = {'first': [], 'second': []}
current_axis_selection = 'first'  # 'first' oder 'second'
# Speichert die bestätigten Achsen
axes_confirmed = {'first': None, 'second': None}

# Funktionen Definitionen


def draw_plane(plotter, points, plane_name='plane', color='yellow'):
    if len(points) < 3:
        return  # Nicht genug Punkte für eine Ebene

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)

    # Ecken des Polygons im 2D-Raum
    min_point = np.min(points_2d, axis=0)
    max_point = np.max(points_2d, axis=0)
    corners = np.array([
        [min_point[0], min_point[1]],
        [max_point[0], min_point[1]],
        [max_point[0], max_point[1]],
        [min_point[0], max_point[1]]
    ])

    corners_3d = pca.inverse_transform(corners)
    plane_mesh = pv.PolyData(corners_3d)
    plane_mesh = plane_mesh.delaunay_2d()

    plotter.add_mesh(plane_mesh, color=color, opacity=0.6,
                     name=plane_name, pickable=False)


def align_and_transform(plotter):
    global mesh_transformed, plane_points, axes_confirmed, mesh

    if not all(axes_confirmed.values()):
        print("Nicht alle Achsen wurden bestätigt.")
        return

    # Berechne die Hauptnormalen für beide ausgewählte Punktmengen
    normals = {}
    centers = {}
    for key, points in plane_points.items():
        if len(points) >= 3:
            pca = PCA(n_components=3).fit(points)
            # Die letzte Komponente ist die Normale
            normals[key] = pca.components_[-1]
            centers[key] = np.mean(points, axis=0)
        else:
            print(f"Nicht genügend Punkte für die Ebene {key}.")
            return

    # Bestimme die Zielnormalen basierend auf den ausgewählten Achsen
    axis_vectors = {'x': np.array([1, 0, 0]), 'y': np.array(
        [0, 1, 0]), 'z': np.array([0, 0, 1])}
    target_normals = {key: axis_vectors[axis]
                      for key, axis in axes_confirmed.items()}

    # Berechne die Rotation von der ersten Normalen zur ersten Zielnormalen
    rotation1 = R.align_vectors(
        [target_normals['first']], [normals['first']])[0]

    # Wende die erste Rotation an und berechne die neue Normale der zweiten Ebene
    rotated_second_points = rotation1.apply(
        plane_points['second'] - centers['first']) + centers['first']
    pca_second_rotated = PCA(n_components=3).fit(rotated_second_points)
    normal_second_rotated = pca_second_rotated.components_[-1]

    # Berechne die zweite Rotation innerhalb der durch die erste Rotation definierten Ebene
    rotation2 = R.align_vectors([target_normals['second']], [
                                normal_second_rotated])[0]

    # Kombiniere beide Rotationen
    combined_rotation = rotation2 * rotation1

    # Wende die kombinierte Rotation auf das Mesh an
    if mesh_transformed is None:
        mesh_transformed = mesh.copy()

    mesh_transformed.points = combined_rotation.apply(
        mesh_transformed.points - centers['first']) + centers['first']

    # Aktualisiere die Anzeige
    # Zurücksetzen der Punkte für die nächste Auswahl
    plane_points = {'first': [], 'second': []}
    # Zurücksetzen der Achsenbestätigung
    axes_confirmed = {'first': None, 'second': None}
    plotter.clear()
    add_mesh(mesh_transformed)
    plotter.render()
    after_render()
    print("Mesh erfolgreich ausgerichtet.")


def save_transformed():
    global mesh_transformed
    if mesh_transformed is not None:
        mesh_transformed.save('transformed_mesh.stl')
        print("Transformed mesh saved.")
    else:
        print("Kein transformiertes Mesh zum Speichern vorhanden.")


def on_pick(event):
    global picked_points, mesh_transformed, plane_points, current_axis_selection
    point = np.array([event[0], event[1], event[2]])

    if any(np.array_equal(point, p) for p in picked_points):
        picked_points = [
            p for p in picked_points if not np.array_equal(point, p)]
    else:
        picked_points.append(point)
        # Speichere den Punkt basierend auf der aktuellen Achsenauswahl
        plane_points[current_axis_selection].append(point)

    plotter.clear()
    if mesh_transformed is not None:
        add_mesh(mesh_transformed)
    else:
        add_mesh(mesh)
    if plane_points['first']:
        draw_plane(plotter, np.array(
            plane_points['first']), 'first_plane', 'yellow')
    if plane_points['second']:
        draw_plane(plotter, np.array(
            plane_points['second']), 'second_plane', 'green')
    plotter.render()
    after_render()


def confirm_axis_selection(plotter, axis):
    global axes_confirmed, current_axis_selection
    if current_axis_selection == 'first':
        axes_confirmed['first'] = axis
        current_axis_selection = 'second'  # Wechsle zur Auswahl der zweiten Achse
    elif current_axis_selection == 'second' and axes_confirmed['first'] != axis:
        axes_confirmed['second'] = axis
    else:
        print("Bitte wählen Sie eine andere Achse als die erste.")
        return

    print(f"Achse {axis.upper()} für {current_axis_selection} bestätigt.")

    # Nach der Bestätigung der zweiten Achse führe die Ausrichtung durch
    if all(axes_confirmed.values()):
        align_and_transform(plotter)


def reset_selection(plotter):
    global picked_points, plane_points, axes_confirmed, current_axis_selection
    picked_points = []
    plane_points = {'first': [], 'second': []}
    axes_confirmed = {'first': None, 'second': None}
    current_axis_selection = 'first'
    plotter.clear()
    add_mesh(mesh)
    plotter.render()
    after_render()
    print("Auswahl zurückgesetzt.")


def remove_last_picked_point(plotter):
    global picked_points, plane_points, current_axis_selection
    if picked_points:
        removed_point = picked_points.pop()
        if removed_point in plane_points[current_axis_selection]:
            plane_points[current_axis_selection].remove(removed_point)
        print(f"Letzter ausgewählter Punkt entfernt: {removed_point}")
        plotter.clear()
        add_mesh(mesh)
        if plane_points['first']:
            draw_plane(plotter, np.array(
                plane_points['first']), 'first_plane', 'yellow')
        if plane_points['second']:
            draw_plane(plotter, np.array(
                plane_points['second']), 'second_plane', 'green')
        plotter.render()
        after_render()
    else:
        print("Keine Punkte zum Entfernen.")


def add_mesh(newMesh):
    plotter.add_mesh(newMesh, pickable=True, show_edges=True)


def after_render():
    # Zeigt Hinweise zur Steuerung an
    plotter.add_text(
        'x/y/z: Confirm axis for current selection\nr: Reset\nb: Remove last picked point\ns: Save Mesh',
        position='lower_right',
        color='black',
        shadow=True,
        font_size=8,
    )


def select_mesh_file():
    root = tk.Tk()
    root.withdraw()  # Verstecke das Tkinter-Hauptfenster
    file_path = filedialog.askopenfilename(
        title="Choose your mesh as .ply file",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
    return file_path


# Auswahl der Mesh-Datei
filename = select_mesh_file()
if filename:
    mesh = pv.read(filename)
else:
    print("No file selected. The program will terminate.")
    exit()

# Einrichten des Plotters
plotter = pv.Plotter()
plotter.show_axes()
add_mesh(mesh)
plotter.enable_point_picking(callback=on_pick, picker='volume')

# Tastenkürzel für die Bestätigung der Achsen
plotter.add_key_event('x', lambda: confirm_axis_selection(plotter, 'x'))
plotter.add_key_event('y', lambda: confirm_axis_selection(plotter, 'y'))
plotter.add_key_event('z', lambda: confirm_axis_selection(plotter, 'z'))

# Tastenkürzel für andere Funktionen
plotter.add_key_event('s', save_transformed)
plotter.add_key_event('r', lambda: reset_selection(plotter))
plotter.add_key_event('b', lambda: remove_last_picked_point(plotter))

# Starte die Visualisierung
plotter.show()
after_render()
