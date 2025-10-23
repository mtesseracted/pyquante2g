import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

def plot_sphere(ax, center, radius, color):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_surface(x, y, z, color=color, shade=True)

def plot_cylinder(ax, point1, point2, radius, color):
    v = np.array(point2) - np.array(point1)
    mag = np.linalg.norm(v)
    v = v / mag

    not_v = np.array([1, 0, 0])
    if np.allclose(v, not_v):
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)

    t = np.linspace(0, 2*np.pi, 20)
    circle = radius * (np.outer(np.cos(t), n1) + np.outer(np.sin(t), n2))

    z = np.linspace(0, mag, 2)
    for i in range(len(circle)):
        x = point1 + circle[i]
        ax.plot([x[0], x[0]+v[0]*mag], [x[1], x[1]+v[1]*mag], [x[2], x[2]+v[2]*mag], color=color)


def plot_molecule_3d(ax, atom_coords, atom_symbols=None, atom_colors=None, bonds=None):
    for ii, coord in enumerate(atom_coords): # Plot atoms
        col = atom_colors[ii] if atom_colors else 'red'
        plot_sphere(ax, coord, radius=0.8, color=col)

    if bonds: # Plot bonds
        for i, j in bonds:
            plot_cylinder(ax, atom_coords[i], atom_coords[j], radius=0.05, color='gray')

    if atom_colors and atom_symbols:
        uniq_atoms = dict(zip(atom_symbols, atom_colors))
        legend_handles = [Patch(color=c, label=n) for n, c in uniq_atoms.items()]
        ax.legend(handles=legend_handles, title="Atom Types")

    return ax


def plot_molecule_3d_zproj(ax, atom_coords, atom_colors=None, bonds=None, zlvl=0.0):
    for coord, color in zip(atom_coords, atom_colors):
        ax.scatter(coord[0], coord[1], zlvl, color=color, alpha=0.3, s=80, edgecolors='k')
    for i, j in bonds:
        x = [atom_coords[i][0], atom_coords[j][0]]
        y = [atom_coords[i][1], atom_coords[j][1]]
        z = [zlvl, zlvl]
        ax.plot(x, y, z, color='gray', alpha=0.3)
    return ax


def plot_molecule_3d_scatter(ax, atom_coords, atom_symbols=None, atom_colors=None, bonds=None):
    sparams = dict(marker='o', edgecolors='dimgray', s=100)
    if atom_colors:
        ax.scatter(*atom_coords.T, c=atom_colors, **sparams)
    else:
        ax.scatter(*atom_coords.T, color='blue', **sparams)

    if bonds: # Plot bonds
        for i, j in bonds:
                x = [atom_coords[i][0], atom_coords[j][0]]
                y = [atom_coords[i][1], atom_coords[j][1]]
                z = [atom_coords[i][2], atom_coords[j][2]]
                ax.plot(x, y, z, color='gray', linewidth=2)

    if atom_colors and atom_symbols:
        uniq_atoms = dict(zip(atom_symbols, atom_colors))
        legend_handles = [Patch(color=c, label=n) for n, c in uniq_atoms.items()]
        ax.legend(handles=legend_handles, title="Atom Types")

    return ax




if __name__ == "__main__":
# Example data
    atom_coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    atom_names = ['H', 'O', 'C']
    atom_colors = ['white', 'red', 'black']
    bonds = [(0, 1), (1, 2)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot atoms
    for coord, color in zip(atom_coords, atom_colors):
        plot_sphere(ax, coord, radius=0.2, color=color)

    # Plot bonds
    for i, j in bonds:
        plot_cylinder(ax, atom_coords[i], atom_coords[j], radius=0.05, color='gray')

    # bb = mol.bbox(padding=0.0).numpy()
    # mx_lim = 0.55 * np.max(bb[:,1] - bb[:,0])
    # bb_cens = 0.5 * (bb[:,1] + bb[:,0])
    # ax.set_xlim([bb_cens[0]-mx_lim, bb_cens[0]+mx_lim])
    # ax.set_ylim([bb_cens[1]-mx_lim, bb_cens[1]+mx_lim])
    # ax.set_zlim([bb_cens[2]-mx_lim, bb_cens[2]+mx_lim])
    ax.set_box_aspect([1,1,1])
    plt.show()

