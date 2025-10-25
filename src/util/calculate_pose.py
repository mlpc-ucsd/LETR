import numpy as np

def get_rotation(pred_lines, centered_points):
    u, s, vh = np.linalg.svd(centered_points)
    # get z direction
    z = vh[2]
    if z[2]>0:
        z = -1*z

    # get x direction
    segment_lengths = []
    for l in pred_lines:
        e1, e2 = l[:3], l[3:]
        segment_lengths.append(np.linalg.norm(e1-e2))

    segment_lengths = np.array(segment_lengths)
    sorti = np.argsort(segment_lengths)

    longest_2 = pred_lines[sorti[-2:]]
    d1 = longest_2[0][:3] - longest_2[0][3:]
    d2 = longest_2[1][:3] - longest_2[1][3:]
    if np.dot(d1, -d2) > 0:
        x = np.mean(np.array([d1, -d2]), axis=0)
    else:
        x = np.mean(np.array([d1, d2]), axis=0)

    # orthogonalize x, z
    z /= np.linalg.norm(z)

    x = x - np.dot(z, x)*z
    x /= np.linalg.norm(x)

    # get y direction
    y = np.linalg.cross(z, x)

    # rotation matrix
    R = np.zeros([3, 3])
    R[:, 0] = x
    R[:, 1] = y
    R[:, 2] = z

    return R

def compute_distances(arr):
    dist_matrix = np.linalg.norm(arr[:, np.newaxis] - arr, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)  # Don't consider distance to self
    return dist_matrix

def merge_closest(arr):
    while len(arr) > 4:
        # Compute pairwise distances
        dist_matrix = compute_distances(arr)

        # Find the indices of the closest points
        idx1, idx2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # Merge the two closest points by replacing them with their mean
        new_point = np.mean([arr[idx1], arr[idx2]], axis=0)

        # Remove the two original points and add the new merged point
        arr = np.delete(arr, [idx1, idx2], axis=0)
        arr = np.vstack([arr, new_point])

    return arr

def get_translation(segment_points, z_direction, bin_height):
    processed = merge_closest(segment_points)
    segments_center = processed.mean(axis=0)
    T = segments_center - z_direction * (bin_height / 2)
    return T

def calculate_pose(pred_lines, pred_scores, bin_height):
    # get 4 line segments with highest score
    keep = np.array(np.argsort(pred_scores)[::-1][:4])
    pred_lines = pred_lines[keep]

    segment_points = pred_lines.reshape(8, 3)
    segments_center = segment_points.mean(axis=0)

    R = get_rotation(pred_lines, segment_points - segments_center)
    T = get_translation(segment_points, R[:, 2], bin_height)

    return R, T
