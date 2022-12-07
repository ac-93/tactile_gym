import numpy as np
from pyrsistent import s
from sklearn.utils import resample
import trimesh
import pybullet as pb

def rotate_vector_by_quaternion(v, q):
    """
    q is the quaternion given by pybullet in the form (x, y, z, w).
    v is the vector to translate. In our case, we want to translate z=[0,0,1],
    which is the normal of the plane on which the TacTip base is placed.
    v_prime is the translated vector
    This is taken from OpenGLM: https://github.com/g-truc/glm/blob/master/glm/detail/type_quat.inl
    Parameters:
        v = vector to rotate
        q = orientation of the TCP (tuple dim 4)
    """
    u = np.array([q[0], q[1], q[2]])
    s = q[3]
    v_prime = v + ((np.cross(u, v) * s) + np.cross(u, np.cross(u, v))) * 2.0
    return v_prime

def create_meshgrid_wrld(c, nx=5, ny=5):
    """
    This function takes the local TCP normal (worldframe) and returns the coordinates of n (by default 25)
    points on a grid defined over the plane perpendicular to vector (0,0,1) and passing through the 
    centre of mass. The dimension of this plane are the sensor's width x depth.
    Parameters:
        c = centre of mass
    Returns:
        numpy array (25, 3) representing the locations of the points on the simple grid wrt worldframe
    """
    x = np.linspace(c[0]-0.015, c[0]+0.015, nx)
    y = np.linspace(c[1]-0.015, c[1]+0.015, ny)
    xv, yv = np.meshgrid(x, y)
    z = np.full(nx * ny, c[2])
    grid_vecs = np.dstack((xv.ravel(),yv.ravel(),z))[0]
    return grid_vecs

def grid_to_TCP_wlrd(c, z_TCP_wrld, nx, ny):
    """
    Rotate all the points in the simple grid to the TCP local frame (wrt worldframe)
    Parameters:
        c = centre of mass in worldframe
        q = orientation of the TCP (tuple dim 4)
    Returns
        the position of the points on the simple grid, np.array (25, 3)
        the position of the points on the transformed grid. These points lie on a plane
            perpendicular to the TCP norm and passing through its centre of mass, np.array(25, 3)
    """
    grid_vecs = create_meshgrid_wrld(c, nx, ny)
    grid_vecs_TCP_wrld = grid_vecs + 0.025*z_TCP_wrld
    return grid_vecs, grid_vecs_TCP_wrld

def shoot_rays(current_TCP_pos_vel_worldframe, pb, nx, ny, draw_rays=False):
    """
    Shoot rays from a plane build around the centre of mass. The plane normal is parallel to the 
    TCP normal.
    Parameters:
        current_TCP_pos_vel_worldframe: returned by robot.arm.get_current_TCP_pos_vel_worldframe()
        pb: pybullet lib to draw vectors
    Return
        grid_vecs: the position of the points on the simple grid around centre of mass, np.array (25, 3)
        grid_vecs_TCP_wrld: the position of the points on the transformed grid. These points lie on a plane
            perpendicular to the TCP norm and passing through its centre of mass, np.array(25, 3)
    """
    # multiple vectors parallel to TCP normal
    # simple grid of points to rotate. It's used to shoot a batch of rays to extract the local shape
    z = np.array([0, 0, 1])
    quaternion_wrld = current_TCP_pos_vel_worldframe[2] # TCP orientation
    z_TCP_wrld = rotate_vector_by_quaternion(z, quaternion_wrld)
    c_wrld = current_TCP_pos_vel_worldframe[0] # TCP centre of mass
    grid_vecs, grid_vecs_TCP_wrld = grid_to_TCP_wlrd(c_wrld, z_TCP_wrld, nx, ny)

    if draw_rays:
        for i in range(0, 25):
            start_point = grid_vecs_TCP_wrld[i] - 0.05 * z_TCP_wrld
            end_point = start_point + 0.025 * z_TCP_wrld
            pb.addUserDebugLine(start_point, end_point, lifeTime=0.05)
    
    return grid_vecs, grid_vecs_TCP_wrld, z_TCP_wrld

def get_contact_points(current_TCP_pos_vel_worldframe, pb, nx=5, ny=5, plot_start_end_rays=False):
    """
    When the sensor touches a surface, returns the contact points along with other info. 
    Parameters:
        current_TCP_pos_vel_worldframe: returned by robot.arm.get_current_TCP_pos_vel_worldframe()
        pb: pybullet lib to draw vectors
    Return
        results: objectUniqueId, linkIndex, hit_fraction, hit_position, hit_normal
    """
    _, grid_vecs_TCP_wrld, z_TCP_wrld = shoot_rays(current_TCP_pos_vel_worldframe, pb, nx, ny)
    # the grid is defined 0.5 units in front of the plane passing through the TCP centrer of mass
    raysFrom = grid_vecs_TCP_wrld - 0.05 * z_TCP_wrld  
    # 0.025 is approx. the distance necessary to cover the entire TCP height + a bit more
    raysTo = raysFrom + 0.025 * z_TCP_wrld

    # shoot rays in batches (the max allowed batch), then put them all together
    max_rays = pb.MAX_RAY_INTERSECTION_BATCH_SIZE
    size = raysFrom.shape[0]
    if size > max_rays:
        results = []
        start = 0
        while end != size:
            end = start + max_rays if (start + max_rays < size) else size
            rays = pb.rayTestBatch(raysFrom[start:end], raysTo[start:end])
            results.append(rays)
            start = start + max_rays
        results = np.array(results, dtype=object)
    else:
        results = np.array(pb.rayTestBatch(raysFrom, raysTo), dtype=object)

    return results


def filter_point_cloud(contact_info):
    """ Receives contact information from PyBullet. It filters out null contact points
    Params:
        contact_info = results of PyBullet pb.rayTestBatch. It is a tuple of objectUniqueId,  linkIndex, hit_fraction, hit_position, hit_normal. Contact info is calculated in robot.blocking_move -> get_contact_points()
    Return:
        filtered point cloud
     """
    # filter out null contacts and convert tuple -> np.array()
    contact_info_non_null = contact_info[contact_info[:,0]!=-1]
    point_cloud = contact_info_non_null[:, 3]
    point_cloud = np.array([np.array(_) for _ in point_cloud])

    return point_cloud


def pointcloud_to_vertices_wrk(point_cloud, robot, args):
    """
    Method to reduce the point cloud obtained from pyBullet into 25 vertices.
    It receives filtered contact information and computes 25 k_means for the non-null contact points. 
    These points represent vertices that are used by the function pointcloud_to_mesh(), which transforms these 25 vertices into a mesh. 
    Vertices are converted to workframe.
    
    Params:
        point_cloud = filtered point cloud, null contact points are not included
    Return:
        mesh = open3d.geometry.TriangleMesh, 25 vertices and faces of the local geometry at touch site
    """
    # compute k-means that will used as vertices
    print(f'Shape of full pointcloud: {point_cloud.shape}')

    verts_wrld = trimesh.points.k_means(point_cloud, 25)[0]

    tcp_pos_wrld, tcp_rpy_wrld, _, _, _ = robot.arm.get_current_TCP_pos_vel_worldframe()

    rot_Q = pb.getQuaternionFromEuler(tcp_rpy_wrld)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    rot_M_inv = np.linalg.inv(rot_M)
    verts_wrk = rot_M_inv @ (verts_wrld - tcp_pos_wrld).transpose(1,0)
    verts_wrk = verts_wrk.transpose(1,0)

    print(f'Point cloud to vertices: {verts_wrk.shape}')

    if args.debug_show_mesh_wrk:
        trimesh.points.plot_points(verts_wrk)

    return verts_wrk
    