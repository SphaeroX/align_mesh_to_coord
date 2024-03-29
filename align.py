# Notwendige Bibliotheken
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog

# Settings
secound_align_repeatments = 10  # Anzahl der Wiederholungen für die zweite Achse

# Globale Variablen
picked_points = []
mesh_transformed = None
plane_points = {'first': [], 'second': []}
current_axis_selection = 'first'  # 'first' oder 'second'
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


def align_first_axis(plotter):
    global mesh_transformed, plane_points, picked_points, axes_confirmed, mesh

    if axes_confirmed['first'] is None:
        print("Die erste Achse wurde nicht bestätigt.")
        return

    points = np.array(plane_points['first'])
    if len(points) >= 3:
        pca = PCA(n_components=3).fit(points)
        normal_first = pca.components_[-1]
        center_first = np.mean(points, axis=0)
    else:
        print("Nicht genügend Punkte für die erste Achse.")
        return

    axis_vectors = {'x': np.array([1, 0, 0]), 'y': np.array(
        [0, 1, 0]), 'z': np.array([0, 0, 1])}
    target_normal_first = axis_vectors[axes_confirmed['first']]

    # Berechne die Rotation
    rotation = R.align_vectors([target_normal_first], [normal_first])[0]

    # Berechne die Verschiebung
    translation = -center_first

    # Wende die Rotation und Verschiebung auf das Mesh an
    if mesh_transformed is None:
        mesh_transformed = mesh.copy()
    mesh_transformed.points = rotation.apply(
        mesh_transformed.points + translation) + center_first

    # Wende die gleiche Transformation auf die plane_points an
    for key in plane_points:
        if plane_points[key]:  # Prüfe, ob die Liste nicht leer ist
            transformed_points = np.array(plane_points[key]) + translation
            transformed_points = rotation.apply(
                transformed_points) + center_first
            plane_points[key] = transformed_points.tolist()

    # Wende die gleiche Transformation auf die picked_points an
    if picked_points:
        transformed_picked_points = np.array(picked_points) + translation
        transformed_picked_points = rotation.apply(
            transformed_picked_points) + center_first
        picked_points = transformed_picked_points.tolist()

    # Aktualisiere die Anzeige
    plotter.clear()
    add_mesh(mesh_transformed)
    plotter.render()
    after_render()
    print("Mesh, plane_points und picked_points erfolgreich an der ersten Achse ausgerichtet.")


def align_second_axis(plotter):
    global mesh_transformed, plane_points, axes_confirmed

    if axes_confirmed['second'] is None:
        print("Die zweite Achse wurde nicht bestätigt.")
        return

    # Konvertiere die Liste in ein numpy Array für die Verarbeitung
    points = np.array(plane_points['second'])
    if len(points) < 3:
        print("Nicht genügend Punkte für die zweite Achse.")
        return

    pca = PCA(n_components=3).fit(points)
    normal_second = pca.components_[-1]
    center_second = np.mean(points, axis=0)

    axis_vectors = {'x': np.array([1, 0, 0]), 'y': np.array(
        [0, 1, 0]), 'z': np.array([0, 0, 1])}
    target_normal_second = axis_vectors[axes_confirmed['second']]

    # Berechne die Rotation
    rotation = R.align_vectors([target_normal_second], [normal_second])[0]

    # Berechne die Verschiebung
    translation = -center_second

    # Wende die Rotation und Verschiebung auf das Mesh an
    mesh_transformed.points = rotation.apply(
        mesh_transformed.points + translation) + center_second

    # Wende die gleiche Transformation auf die plane_points an
    for key in plane_points:
        if plane_points[key]:  # Prüfe, ob die Liste nicht leer ist
            transformed_points = np.array(
                plane_points[key]) + translation  # Verschiebung anwenden
            transformed_points = rotation.apply(
                transformed_points) + center_second  # Rotation anwenden
            # Konvertiere das Array zurück in eine Liste
            plane_points[key] = transformed_points.tolist()

    # Aktualisiere die Anzeige
    plotter.clear()
    add_mesh(mesh_transformed)
    plotter.render()
    after_render()
    print("Mesh und plane_points erfolgreich an der zweiten Achse ausgerichtet.")


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

    # Prüfe, ob der Punkt bereits ausgewählt wurde, und entferne ihn in diesem Fall
    # oder füge ihn hinzu, wenn er noch nicht ausgewählt wurde
    if any(np.array_equal(point, p) for p in picked_points):
        picked_points = [
            p for p in picked_points if not np.array_equal(point, p)]
        if point.tolist() in plane_points[current_axis_selection]:
            plane_points[current_axis_selection].remove(point.tolist())
    else:
        picked_points.append(point)
        # Speichere den Punkt basierend auf der aktuellen Achsenauswahl
        plane_points[current_axis_selection].append(point.tolist())

    # Lösche die aktuelle Darstellung, um sie mit den aktualisierten Informationen neu zu zeichnen
    plotter.clear()

    # Füge das transformierte oder das ursprüngliche Mesh hinzu, abhängig vom Zustand
    if mesh_transformed is not None:
        add_mesh(mesh_transformed)
    else:
        add_mesh(mesh)

    # Zeichne die Ebenen für die erste und zweite Achsenauswahl
    if plane_points['first']:
        draw_plane(plotter, np.array(
            plane_points['first']), 'first_plane', 'yellow')
    if plane_points['second']:
        draw_plane(plotter, np.array(
            plane_points['second']), 'second_plane', 'green')

    # Zeige die ausgewählten Punkte in Rot
    if picked_points:
        plotter.add_mesh(pv.PolyData(np.array(picked_points)),
                         color="red", point_size=10, render_points_as_spheres=True)

    # Führe erforderliche Aktualisierungen der Darstellung durch
    plotter.render()
    after_render()


def confirm_axis_selection(plotter, axis):
    global axes_confirmed, current_axis_selection
    if current_axis_selection == 'first':
        axes_confirmed['first'] = axis
        # Rufe die Ausrichtung für die erste Achse direkt hier auf
        align_first_axis(plotter)
        current_axis_selection = 'second'  # Wechsle zur Auswahl der zweiten Achse
    elif current_axis_selection == 'second' and axes_confirmed['first'] != axis:
        axes_confirmed['second'] = axis
        # Optionale Ausrichtung für die zweite Achse
        for _ in range(secound_align_repeatments):
            align_second_axis(plotter)
            align_first_axis(plotter)

        reset_selection(plotter)
    else:
        print("Bitte wählen Sie eine andere Achse als die erste.")


def reset_selection(plotter):
    global picked_points, mesh_transformed, plane_points, axes_confirmed, current_axis_selection
    picked_points = []
    plane_points = {'first': [], 'second': []}
    axes_confirmed = {'first': None, 'second': None}
    current_axis_selection = 'first'

    plotter.clear()  # Lösche alle vorherigen Meshes aus dem Plotter
    add_mesh(mesh_transformed)  # Füge das ursprüngliche Mesh erneut hinzu

    plotter.render()
    after_render()
    print("Auswahl und Mesh wurden zurückgesetzt.")


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
