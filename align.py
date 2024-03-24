import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

# Globale Variablen
picked_points = []
mesh_transformed = None

# Laden des Meshes aus der Datei
filename = 'input_mesh.ply'
mesh = pv.read(filename)

# Funktionen Definitionen


def draw_plane(plotter, points):
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

    plotter.add_mesh(plane_mesh, color='yellow', opacity=0.6,
                     name='plane', pickable=False)


def align_and_transform(plotter, target_axis):
    global picked_points, mesh_transformed
    if len(picked_points) < 3:
        print("Nicht genügend Punkte für die Transformation.")
        return

    current_mesh = mesh_transformed if mesh_transformed is not None else mesh

    # Schwerpunkt und Normale der ausgewählten Punkte berechnen
    center_of_mass = np.mean(picked_points, axis=0)
    pca = PCA(n_components=3)
    pca.fit(picked_points)
    normal = pca.components_[2]  # Die Normale der Ebene

    # Zielnormalen definieren
    target_normal = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }[target_axis]

    # Rotation bestimmen, um die aktuelle auf die Zielnormale abzubilden
    rotation = R.align_vectors([target_normal], [normal])[0]
    rotation_matrix = rotation.as_matrix()

    # Transformation des Meshes
    mesh_centered = current_mesh.points - center_of_mass
    points_transformed = np.dot(
        rotation_matrix, mesh_centered.T).T + center_of_mass
    mesh_transformed = current_mesh.copy()
    mesh_transformed.points = points_transformed

    # Aktualisiere die Anzeige
    picked_points = []
    plotter.clear()
    plotter.add_mesh(mesh_transformed, pickable=True)
    plotter.render()
    after_render()
    print(f"Alignment auf die {target_axis.upper()}-Achse durchgeführt.")


def save_transformed():
    global mesh_transformed
    if mesh_transformed is not None:
        mesh_transformed.save('transformed_mesh.stl')
        # Extrahiere die Punktwolke aus dem Mesh und speichere sie
        print("Transformed mesh saved.")
    else:
        print("Kein transformiertes Mesh zum Speichern vorhanden.")


def on_pick(event):
    # Zugriff auf die globale Variable `mesh_transformed`
    global picked_points, mesh_transformed
    point = np.array([event[0], event[1], event[2]])
    threshold = 0.001  # Distanzschwelle für die Punktauswahl

    # Näheprüfung zu bereits ausgewählten Punkten
    for i, p in enumerate(picked_points):
        if np.linalg.norm(point - p) < threshold:
            picked_points.pop(i)  # Entferne den Punkt, wenn er nahe genug ist
            break
    else:
        # Füge den Punkt hinzu, wenn er nicht zu nahe ist
        picked_points.append(point)

    plotter.clear()  # Lösche die aktuelle Darstellung
    # Prüfe, ob eine Transformation durchgeführt wurde und füge das entsprechende Mesh hinzu
    if mesh_transformed is not None:
        # Zeige das transformierte Mesh
        plotter.add_mesh(mesh_transformed, pickable=True)
    else:
        # Zeige das ursprüngliche Mesh, falls keine Transformation stattgefunden hat
        plotter.add_mesh(mesh, pickable=True)
    if picked_points:
        # Zeige ausgewählte Punkte, falls vorhanden
        plotter.add_mesh(pv.PolyData(np.array(picked_points)),
                         color="red", point_size=10)
    # Zeichne die Ebene basierend auf den ausgewählten Punkten
    draw_plane(plotter, np.array(picked_points))
    plotter.render()
    after_render()


def reset_selection(plotter):
    global picked_points
    picked_points = []
    plotter.remove_actor('plane')
    plotter.render()
    after_render()
    print("Selektionen zurückgesetzt und Ebene entfernt.")


def remove_last_picked_point(plotter):
    if picked_points:
        removed_point = picked_points.pop()
        print(f"Last picked point removed: {removed_point}")
        plotter.clear()
        plotter.add_mesh(mesh, pickable=True)
        draw_plane(plotter, np.array(picked_points))
        plotter.add_mesh(pv.PolyData(np.array(picked_points)),
                         color="red", point_size=10)
        plotter.render()
        after_render()
    else:
        print("No points to remove.")


def after_render():
    plotter.add_text(
        'x/y/z = align plane  to axis\ns = save\nr = reset all points\nb: remove the last clicked',
        position='lower_right',
        color='black',
        shadow=True,
        font_size=8,
    )
    # plotter.add_camera_orientation_widget()


# Einrichten des Plotters
plotter = pv.Plotter()
plotter.show_axes()
plotter.add_mesh(mesh, pickable=True)

# plotter.show_bounds(  grid='front',  location='outer', all_edges=True)
plotter.enable_point_picking(callback=on_pick)

# Tastenkürzel für die Transformation auf die x-, y- und z-Achse
plotter.add_key_event('x', lambda: align_and_transform(plotter, 'x'))
plotter.add_key_event('y', lambda: align_and_transform(plotter, 'y'))
plotter.add_key_event('z', lambda: align_and_transform(plotter, 'z'))

# Tastenkürzel zum Speichern der transformierten Punktwolke
plotter.add_key_event('s', save_transformed)

# Tastenkürzel zum Zurücksetzen aller Selektionen und Löschen der Ebene
plotter.add_key_event('r', lambda: reset_selection(plotter))

# Tastenkürzel zum Entfernen des zuletzt ausgewählten Punktes
plotter.add_key_event('b', lambda: remove_last_picked_point(plotter))

# Starte die Visualisierung
plotter.show()
after_render()
