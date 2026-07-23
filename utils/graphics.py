import numpy as np
import cv2

dragging = False
xi, yi = -1, -1


def rotation_matrix(theta):
    """Rotation matrix from Euler angles.

    Args:
        theta: the (roll, pitch, yaw) triple in radians.
    Returns:
        The (3, 3) matrix, composed Rz @ Ry @ Rx.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


class Camera:
    """Orbit camera for the wireframe viewer: it turns around a centre point
    and projects world points into the image with OpenCV."""

    def __init__(self, pos, theta, cameraMatrix, distCoeffs):
        """Place the camera.

        Args:
            pos: its position in the world frame.
            theta: its (roll, pitch, yaw) in radians.
            cameraMatrix: the OpenCV intrinsic matrix.
            distCoeffs: the OpenCV distortion coefficients.
        Returns:
            Nothing.
        """
        self.pos = pos                          # wrt world frame
        self.theta = theta                      # Euler angles: roll pitch yaw
        self.rMat = rotation_matrix(theta)

        self.center = np.zeros(3)               # camera rotates around center
        self.r = np.array([-8., 0., 0.])

        # intrinsic camera parameters
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    def set_center(self, vector):
        """Move the point the camera orbits -- following a drone means moving
        this to its position every frame.

        Args:
            vector: the new centre in world coordinates.
        Returns:
            Nothing; the camera position is recomputed from it.
        """
        self.center = vector
        self.pos = np.dot(self.rMat, self.r) + self.center

    def rotate(self, theta):
        """Turn the camera around its centre.

        Args:
            theta: the (roll, pitch, yaw) increment in radians.
        Returns:
            Nothing; orientation and position are both updated.
        """
        self.theta += theta
        self.rMat = rotation_matrix(self.theta)
        self.pos = np.dot(self.rMat, self.r) + self.center

    def zoom(self, scl):
        """Move the camera along its orbit radius.

        Args:
            scl: multiplier on the radius -- above 1 pulls back, below 1 moves in.
        Returns:
            Nothing.
        """
        self.r *= scl
        self.pos = self.rMat @ self.r + self.center

    def project(self, points):
        """Project world points onto the image.

        Args:
            points: (n, 3) world coordinates.
        Returns:
            (projected pixels, in_frame) -- the boolean flags which points sit
            in front of the camera, the projection of the others being
            meaningless.
        """
        in_frame = np.dot(points - self.pos, self.rMat[:, 0]) > 0.01

        # x-axis is used as projection axis
        M = np.dot(self.rMat, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))

        tvec = -np.dot(np.transpose(M), self.pos)
        rvec = cv2.Rodrigues(np.transpose(M))[0]

        projected_points = cv2.projectPoints(points, rvec, tvec, self.cameraMatrix, self.distCoeffs)[0].astype(np.int64)
        return projected_points, in_frame

    def mouse_control(self, event, x, y, flags, params):
        """OpenCV mouse callback: dragging turns the camera.

        Args:
            event: the OpenCV event code.
            x, y: cursor position in the window.
            flags: OpenCV flags (unused).
            params: OpenCV user data (unused).
        Returns:
            Nothing; a drag rotates the camera by the cursor travel.
        """
        global xi, yi, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            xi, yi = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                yaw = 2*np.pi * (x - xi) / 1536
                pitch = -np.pi * (y - yi) / 864
                self.rotate([0, pitch, yaw])
                xi, yi = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
#         elif event == cv2.EVENT_MOUSEWHEEL:
#             if flags < 0:
#                 self.r *= 1.05
#                 self.pos = np.dot(self.rMat, self.r) + self.center
#             elif flags > 0:
#                 self.r /= 1.05
#                 self.pos = np.dot(self.rMat, self.r) + self.center


class Mesh:
    """A wireframe: vertices plus the edges joining them."""

    def __init__(self, vertices, edges):
        """Hold the geometry.

        Args:
            vertices: (n, 3) world coordinates.
            edges: pairs of vertex indices to draw a segment between.
        Returns:
            Nothing.
        """
        self.vertices = vertices
        self.edges = edges
        self.pos = np.array([0., 0., 0.])
        self.theta = np.array([0., 0., 0.])

    def draw(self, img, cam, color=(100, 100, 100), pt=1, arrow=False):
        """Draw the wireframe on an image.

        Args:
            img: the image to draw on, modified in place.
            cam: the camera doing the projection.
            color: BGR colour.
            pt: line thickness.
            arrow: draw arrow heads instead of plain segments.
        Returns:
            Nothing; an edge with a vertex behind the camera is skipped.
        """
        pvertices, in_frame = cam.project(self.vertices)
        for edge in self.edges:
            if in_frame[edge[0]] and in_frame[edge[1]]:
                pt1 = tuple(pvertices[edge[0]][0])
                pt2 = tuple(pvertices[edge[1]][0])
                if arrow:
                    cv2.arrowedLine(img, pt1, pt2, color, pt)
                else:
                    cv2.line(img, pt1, pt2, color, pt)

    def translate(self, vector):
        """Move the whole mesh.

        Args:
            vector: the world-frame displacement.
        Returns:
            Nothing; the vertices are moved in place.
        """
        self.pos += vector
        for vertex in self.vertices:
            vertex += vector

    def rotate(self, theta):
        """Set the mesh attitude, turning it about its own position.

        Args:
            theta: the target (roll, pitch, yaw) -- absolute, not an
                increment: the current attitude is undone first.
        Returns:
            Nothing; the vertices are moved in place.
        """
        M1 = np.transpose(rotation_matrix(self.theta))
        M2 = rotation_matrix(theta)
        R = np.dot(M2, M1)
        for vertex in self.vertices:
            delta = self.pos + np.dot(R, vertex - self.pos) - vertex
            vertex += delta
        self.theta = theta


class Force:
    """An arrow drawn at a point of the mesh: one rotor's thrust."""

    def __init__(self, vertex):
        """Attach the arrow.

        Args:
            vertex: where it starts, usually a rotor centre.
        Returns:
            Nothing; the force itself starts at zero (see set_thrust).
        """
        self.vertex = vertex
        self.F = np.zeros(3)

    def draw(self, img, cam, color=(0, 0, 255), pt=1):
        """Draw the arrow.

        Args:
            img: the image to draw on, modified in place.
            cam: the camera doing the projection.
            color: BGR colour.
            pt: line thickness.
        Returns:
            Nothing.
        """
        pt1, _ = cam.project(np.array([self.vertex]))
        pt2, _ = cam.project(np.array([self.vertex + self.F]))
        pt1 = tuple(pt1[0][0])
        pt2 = tuple(pt2[0][0])
        cv2.arrowedLine(img, pt1, pt2, color, pt)


def create_grid(rows, cols, length):
    """Build the ground grid the scene is read against.

    Args:
        rows, cols: number of cells in each direction.
        length: cell size.
    Returns:
        A Mesh centred on the origin, lying in the z = 0 plane.
    """
    rows, cols = rows+1, cols+1     # extra vertex in each direction
    vertices = np.zeros([rows * cols, 3])
    edges = []
    for i in range(rows):
        for j in range(cols):
            vertices[i * cols + j] = [
                i * length - (rows - 1) * length / 2,
                j * length - (cols - 1) * length / 2,
                0.
            ]
            if i != 0:
                edges.append((cols * (i - 1) + j, cols * i + j))
            if j != 0:
                edges.append((cols * i + j - 1, cols * i + j))
    return Mesh(vertices, np.array(edges))


def create_path(vertices, loop=False):
    """Build a polyline through given points -- a trajectory, or the side of a
    shape.

    Args:
        vertices: the points, in order.
        loop: also join the last point back to the first.
    Returns:
        The Mesh.
    """
    edges = [(i, i+1) for i in range(len(vertices)-1)]
    if loop:
        edges.append((0, len(vertices)-1))
    return Mesh(np.array(vertices), np.array(edges))


def create_circle(r, px, py, pz, num=20):
    """Build a horizontal circle, used for the rotor discs.

    Args:
        r: radius.
        px, py, pz: its centre.
        num: how many segments approximate it.
    Returns:
        The Mesh, closed into a loop.
    """
    vertices = np.array([[
        px + r * np.cos(i * 2 * np.pi / num),
        py + r * np.sin(i * 2 * np.pi / num),
        pz
    ] for i in range(num)])
    return create_path(vertices, loop=True)


def group(mesh_list):
    """Merge several meshes into one, so they move and draw together.

    Args:
        mesh_list: the meshes to merge.
    Returns:
        A single Mesh; each part's edge indices are shifted by the number of
        vertices already merged, which is what keeps the edges pointing at the
        right vertices.
    """
    vertices = np.concatenate([
        mesh.vertices for mesh in mesh_list
    ])
    index_shifts = np.cumsum(
        [0] + [len(mesh.vertices) for mesh in mesh_list][:-1]
    )
    edges = np.concatenate([
        mesh.edges + shift for (mesh, shift) in zip(mesh_list, index_shifts)
    ])
    return Mesh(vertices, edges)


def create_drone(r):
    """Build the quadrotor model: four rotor discs, the arms, the body, and a
    nose marker that shows which way it faces.

    Args:
        r: half the arm span, i.e. the overall size.
    Returns:
        (drone mesh, the four Force arrows) -- the last four vertices of the
        mesh are the rotor centres, which is where the arrows attach.
    """
    c1 = create_circle(2*r/3, r, -r, 0.)
    c2 = create_circle(2*r/3, -r, -r, 0.)
    c3 = create_circle(2*r/3, r, r, 0.)
    c4 = create_circle(2*r/3, -r, r, 0.)

    l1 = create_path(np.array([[ 2*r/4,  r/3, r/10], [ r, r, 0.]]))
    l2 = create_path(np.array([[ 2*r/4, -r/3, r/10], [ r,-r, 0.]]))
    l3 = create_path(np.array([[-2*r/4, -r/3, r/10], [-r,-r, 0.]]))
    l4 = create_path(np.array([[-2*r/4,  r/3, r/10], [-r, r, 0.]]))
    
    box = create_path(np.array([
        [ 2*r/4,  r/3, r/10],
        [ 2*r/4, -r/3, r/10],
        [-2*r/4, -r/3, r/10],
        [-2*r/4,  r/3, r/10]
    ]), loop=True)
    
    l5 = create_path(np.array([
        [ 2*r/4,          r/3, r/10],
        [ 2*r/4+r/3,  0.7*r/3, r/10],
        [ 2*r/4+r/3, -0.7*r/3, r/10],
        [ 2*r/4,         -r/3, r/10]
    ]))
    
    drone = group([c1, c2, c3, c4, l1, l2, l3, l4, l5, box])
    drone.vertices = np.concatenate([
        drone.vertices,
        np.array([[r, -r, 0.], [r, r, 0.], [-r, r, 0.], [-r, -r, 0.]])  # centers of the circles
    ])

    T1, T2, T3, T4 = drone.vertices[-4:]    # thrust on 4 positions
    #Fg = Force(drone.pos)                   # gravity acts on center of mass
    forces = Force(T1), Force(T2), Force(T3), Force(T4)  #, Fg
    return drone, forces


def set_thrust(drone, forces, T):
    """Point the four thrust arrows for the current attitude.

    Args:
        drone: the drone mesh, read for its attitude.
        forces: its four Force arrows.
        T: the four thrust magnitudes.
    Returns:
        Nothing; each arrow is set along the drone's own vertical axis, so they
        tilt with it.
    """
    T1, T2, T3, T4 = forces
    T1.F = - T[0] * rotation_matrix(drone.theta)[:, 2]
    T2.F = - T[1] * rotation_matrix(drone.theta)[:, 2]
    T3.F = - T[2] * rotation_matrix(drone.theta)[:, 2]
    T4.F = - T[3] * rotation_matrix(drone.theta)[:, 2]

