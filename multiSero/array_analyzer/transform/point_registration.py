import cv2 as cv
import logging
import numpy as np

import multiSero.array_analyzer.extract.constants as constants


def icp(source, target, max_iterate=50, matrix_diff=1.):
    """
    Iterative closest point. Expects x, y coordinates of source and target in
    an array with shape: nbr of points x 2

    :param np.array source: Source spot coordinates
    :param np.array target: Target spot coordinates
    :param int max_iterate: Maximum number of registration iterations
    :param float matrix_diff: Sum of absolute differences between transformation
        matrices after one iteration
    :return np.array t_matrix: 2D transformation matrix (2 x 3)
    """
    src = source.copy().astype(np.float32)
    dst = target.copy().astype(np.float32)

    src = np.expand_dims(src, 0)
    dst = np.expand_dims(dst, 0)

    # Initialize kNN module
    knn = cv.ml.KNearest_create()
    labels = np.array(range(dst.shape[1])).astype(np.float32)
    knn.train(dst[0], cv.ml.ROW_SAMPLE, labels)
    # Initialize transformation matrix
    t_matrix = np.eye(3)
    t_temp = np.eye(3)
    t_old = t_matrix

    # Iterate while matrix difference > threshold
    for i in range(max_iterate):

        # Find closest points
        ret, results, neighbors, dist = knn.findNearest(src[0], 1)
        # Outlier removal
        idxs = np.squeeze(neighbors.astype(np.uint8))
        dist_max = 2 * np.median(dist)
        normal_idxs = np.where(dist < dist_max)[0]
        idxs = idxs[normal_idxs]
        # Find rigid transform
        t_iter = cv.estimateRigidTransform(
            src[0, normal_idxs, :],
            dst[0, idxs, :],
            fullAffine=False,
        )
        if t_iter is None:
            print("ICP optimization failed.")
            return None
        t_temp[:2] = t_iter
        src = cv.transform(src, t_iter)
        t_matrix = np.dot(t_temp, t_matrix)
        # Estimate diff
        t_diff = sum(sum(abs(t_matrix[:2] - t_old[:2])))
        t_old = t_matrix

        if t_diff < matrix_diff:
            break

    return t_matrix[:2]


class ParticleFilter:
    """
    Framework for registering grid points to spot coordinates using a
    particle filter approach.
    """
    def __init__(self, spot_coords, im_shape, fiducials_idx, random_seed=None):
        """
        Initialize by creating grid coordinates and particles.

        :param np.array spot_coords: Coordinates of detected spots (nbr spots x 2)
        :param tuple im_shape: Image shape
        :param list fiducials_idx: Indices of grid coordinates which are considered
            fiducials
        :param int random_seed: Optional random seed for deterministic runs
        """
        self.logger = logging.getLogger(constants.LOG_NAME)
        self.im_shape = im_shape
        self.fiducials_idx = fiducials_idx
        # Initialize random number generator
        np.random.seed(random_seed)

        self.spot_coords = spot_coords
        self.grid_coords = self.create_reference_grid()
        self.fiducial_coords = self.grid_coords[self.fiducials_idx, :]
        self.registered_coords = None
        self.registration_ok = True
        self.registered_dist = None
        self.standard_devs = np.array(constants.STDS)
        self.nbr_particles = constants.NBR_PARTICLES
        self.t_matrix = None
        self.mean_point = constants.MEAN_POINT
        self.scale_mean = constants.SCALE_MEAN
        self.angle_mean = constants.ANGLE_MEAN
        self.particles = self.create_gaussian_particles()

    def create_reference_grid(self):
        """
        Generate initial spot grid based on image scale, center point, spot distance
        and spot layout (nbr rows and cols).

        :return np.array grid_coords: (row, col) coordinates for reference spots (nbr x 2)
        """
        nbr_grid_rows = constants.params['rows']
        nbr_grid_cols = constants.params['columns']
        # Distance between spots in pixels
        spot_dist = constants.SPOT_DIST_PIX
        # Assume center point is the center of image as starting point
        center_point = tuple((self.im_shape[0] / 2, self.im_shape[1] / 2))
        start_row = center_point[0] - spot_dist * (nbr_grid_rows - 1) / 2
        start_col = center_point[1] - spot_dist * (nbr_grid_cols - 1) / 2
        row_vals = np.linspace(
            start_row,
            start_row + (nbr_grid_rows - 1) * spot_dist,
            nbr_grid_rows,
        )
        col_vals = np.linspace(
            start_col,
            start_col + (nbr_grid_cols - 1) * spot_dist,
            nbr_grid_cols,
        )
        grid_cols, grid_rows = np.meshgrid(col_vals, row_vals)
        grid_cols = grid_cols.flatten()
        grid_rows = grid_rows.flatten()
        grid_coords = np.vstack([grid_rows.T, grid_cols.T]).T
        return grid_coords

    def create_gaussian_particles(self):
        """
        Create particles from parameters x, y, scale and angle given mean and std.
        A particle is considered one set of parameters for a 2D translation matrix.
        Standard deviations of parameters x, y, angle and scale are predetermined
        and set in constants

        :param tuple mean_point: Mean offset in x and y
        :param float scale_mean: Mean scale of translation
        :param float angle_mean: Mean angle of translation estimation

        :return np.array particles: Set of particle coordinates (nbr particles x 4)
        """
        particles = np.empty((self.nbr_particles, 4))
        particles[:, 0] = self.mean_point[0] +\
                          (np.random.randn(self.nbr_particles) * self.standard_devs[0])
        particles[:, 1] = self.mean_point[1] +\
                          (np.random.randn(self.nbr_particles) * self.standard_devs[1])
        particles[:, 2] = self.angle_mean +\
                          (np.random.randn(self.nbr_particles) * self.standard_devs[2])
        particles[:, 3] = self.scale_mean +\
                          (np.random.randn(self.nbr_particles) * self.standard_devs[3])
        return particles

    @staticmethod
    def get_translation_matrix(particle):
        """
        Create a 2D translation matrix from x, y, scale and angle.

        :param np.array particle: The four parameters x, y, scale and angle
        :return np.array t_matrix: 2D translation matrix (2 x 3)
        """
        a = particle[3] * np.cos(particle[2] * np.pi / 180)
        b = particle[3] * np.sin(particle[2] * np.pi / 180)
        t_matrix = np.array([[a, b, particle[0]],
                            [-b, a, particle[1]]])
        #t_matrix = np.array([[0, 0, particle[0]],
                           #[0, 0, particle[1]]])
        return t_matrix

    def particle_filter(self,
                        max_iter=1000,
                        stop_criteria=.1,
                        iter_decrease=.8,
                        nbr_outliers=0):
        """
        Particle filtering to determine best grid location.
        Start with a number of randomly placed particles. Compute distances
        to nearest neighbors among detected spots. Do importance sampling of
        the particles and slightly distort them while iterating until convergence.
        registered_dist is the minimum sum of distances between registered
        coordinates and spot coordinates, divided by the number of registered
        points.

        :param int max_iter: Maximum number of iterations
        :param float stop_criteria: Absolute difference of distance between iterations
        :param float iter_decrease: Reduce standard deviations each iterations to slow
            down permutations
        :param int nbr_outliers: If registration hasn't converged, remove worst fitted
            spots when running particle filter. Maximum nbr_outliers allowed is
            min(n(fiducial) - 2, n(spots) -2)
        """
        # Use kNN module to petrain spot coords
        dst = self.spot_coords.copy().astype(np.float32)
        knn = cv.ml.KNearest_create()
        labels = np.array(range(dst.shape[0])).astype(np.float32)
        knn.train(dst, cv.ml.ROW_SAMPLE, labels)
        # Make sure we don't have too many outliers
        if nbr_outliers > 0:
            if min(len(labels) - nbr_outliers, self.fiducial_coords.shape[0] - nbr_outliers) < 2:
                nbr_outliers = min(len(labels) - 2, self.fiducial_coords.shape[0] - 2)
        self.logger.debug(
            "Particle filter, number of outliers: {}".format(nbr_outliers),
        )
        dists = np.zeros(self.nbr_particles)
        temp_stds = self.standard_devs.copy()
        temp_particles = self.particles.copy()

        # Iterate until min dist doesn't change
        min_dist_old = 10 ** 6
        for i in range(max_iter):

            for p in range(self.nbr_particles):
                particle = temp_particles[p]
                # Generate transformation matrix
                t_matrix = self.get_translation_matrix(particle)
                trans_coords = cv.transform(np.array([self.fiducial_coords]), t_matrix)
                trans_coords = trans_coords[0].astype(np.float32)
                # Find nearest spots
                ret, results, neighbors, dist = knn.findNearest(trans_coords, 1)
                if nbr_outliers > 0:
                    # Remove worst fitted spots
                    dist = np.sort(dist, axis=0)
                    dist = dist[:-nbr_outliers]

                dists[p] = sum(dist)

            min_dist = np.min(dists)
            self.logger.debug("Iteration: {} min dist: {}".format(i, min_dist))
            # See if min dist is not decreasing anymore
            if abs(min_dist_old - min_dist) < stop_criteria:
                break
            min_dist_old = min_dist

            # Low distance should correspond to high probability
            weights = 1 / dists
            # Make weights sum to 1
            weights = weights / sum(weights)

            # Importance sampling
            idxs = np.random.choice(self.nbr_particles, self.nbr_particles, p=weights)
            temp_particles = temp_particles[idxs, :]

            # Reduce standard deviations a little every iteration
            temp_stds = temp_stds * iter_decrease ** i
            # Distort particles
            for c in range(4):
                distort = np.random.randn(self.nbr_particles)
                temp_particles[:, c] = temp_particles[:, c] + distort * temp_stds[c]

        # Get best particle in terms of nearest to spots
        particle = temp_particles[dists == dists.min(), :][0]

        # Generate transformation matrix
        self.t_matrix = self.get_translation_matrix(particle)
        self.registered_dist = min_dist / (self.fiducial_coords.shape[0] - nbr_outliers)
        self.logger.info("Particle filter min dist: {}".format(self.registered_dist))
        if self.registered_dist > constants.REG_DIST_THRESH:
        #if self.registered_dist > 300:
            self.registration_ok = False
        else:
            self.registration_ok = True
        self.logger.info("Is registration ok: {}".format(self.registration_ok))

    def compute_registered_coords(self):
        """
        Given initial grid coordinates and transformation matrix, compute
        registered grid coordinates
        :return np.array registered_coords: Registered grid coordinates
        """
        assert self.t_matrix is not None,\
            "Transformation matrix not computed"
        self.registered_coords = np.squeeze(
            cv.transform(np.array([self.grid_coords]), self.t_matrix),
        )
        return self.registered_coords

    def check_reg_coords(self):
        """
        Checks that all registered coordinates are within image bounds.

        :param np.array reg_coords: Registered grid coordinates (nbr spots x 2)
        :param tuple im_shape: Image shape (rows, cols)
        :return bool registration_ok: Variable determining if registration is okay
            given image boundaries
        """
        # If registration is already deemed not ok, do nothing
        if self.registration_ok:
            reg_coord_max = np.max(self.registered_coords, axis=0)
            if reg_coord_max[0] >= self.im_shape[0] or reg_coord_max[1] >= self.im_shape[1]:
                self.registration_ok = False
            reg_coord_min = np.min(self.registered_coords, axis=0)
            if np.any(reg_coord_min <= 0):
                self.registration_ok = False
        return self.registration_ok
