import numpy as np
import pytest

import multiSero.array_analyzer.extract.constants as constants
import multiSero.array_analyzer.transform.point_registration as registration


@pytest.fixture
def register_inst():
    im_shape = (50, 100)
    spot_coords = np.array(
        [[18, 51], [10, 10], [16, 20], [22, 20], [29, 41], [31, 59]],
        ).astype(np.float32)
    fiducials_idx = [1, 3, 5]
    constants.params = {
        'rows': 2,
        'columns': 3,
    }
    constants.SPOT_DIST_PIX = 10
    constants.NBR_PARTICLES = 100
    constants.STDS = [1, 1, 1, 1]

    register_inst = registration.ParticleFilter(
        spot_coords=spot_coords,
        im_shape=im_shape,
        fiducials_idx=fiducials_idx,
        random_seed=42,
    )
    return register_inst


def test_create_reference_grid(register_inst):
    grid_coords = register_inst.create_reference_grid()
    row_vals = [20, 30]
    col_vals = np.arange(40, 70, 10)
    count = 0
    for row in row_vals:
        for col in col_vals:
            assert grid_coords[count, 0] == row
            assert grid_coords[count, 1] == col
            count += 1


def test_create_gaussian_particles(register_inst):
    particles = register_inst.create_gaussian_particles()
    assert particles.shape == (100, 4)
    particle_means = np.mean(particles, 0)
    assert -.5 < particle_means[0] < .5
    assert -.5 < particle_means[1] < .5
    assert -.5 < particle_means[2] < .5
    assert .5 < particle_means[3] < 1.5


def test_get_translation_matrix(register_inst):
    particle = [20, 50, 90, 2]
    t_matrix = register_inst.get_translation_matrix(particle)
    assert t_matrix.shape == (2, 3)
    assert t_matrix[0, 0] == t_matrix[1, 1]
    assert t_matrix[0, 1] == -t_matrix[1, 0]
    assert t_matrix[0, 2] == particle[0]
    assert t_matrix[1, 2] == particle[1]
    assert t_matrix[0, 0] < .000001
    assert t_matrix[0, 1] == 2


def test_particle_filter(register_inst):
    register_inst.particle_filter(max_iter=5)
    assert 3.5 < register_inst.registered_dist < 4
    assert register_inst.registration_ok


def test_particle_filter_many_outliers(register_inst):
    register_inst.particle_filter(max_iter=10, nbr_outliers=5)
    # More iterations and 1 outlier leads to better registration
    assert 0 < register_inst.registered_dist < .5
    assert register_inst.registration_ok


def test_compute_registered_coords(register_inst):
    register_inst.t_matrix = np.eye(2, 3)
    register_inst.t_matrix[:, 2] = [5, 10]
    registered_coords = register_inst.compute_registered_coords()
    row_vals = [25, 35]
    col_vals = np.arange(50, 80, 10)
    count = 0
    for row in row_vals:
        for col in col_vals:
            assert registered_coords[count, 0] == row
            assert registered_coords[count, 1] == col
            count += 1


def test_check_reg_coords(register_inst):
    reg_coords = np.ones((5, 2), dtype=np.float32)
    reg_coords[2, :] = [24., 92.1]
    register_inst.registered_coords = reg_coords
    reg_ok = register_inst.check_reg_coords()
    assert reg_ok is True


def test_check_reg_coords_large_col(register_inst):
    reg_coords = np.ones((5, 2), dtype=np.float32)
    reg_coords[2, :] = [24., 100.1]
    register_inst.registered_coords = reg_coords
    reg_ok = register_inst.check_reg_coords()
    assert reg_ok is False


def test_check_reg_coords_large_row(register_inst):
    reg_coords = np.ones((5, 2), dtype=np.float32)
    reg_coords[2, :] = [50., 92.1]
    register_inst.registered_coords = reg_coords
    reg_ok = register_inst.check_reg_coords()
    assert reg_ok is False


def test_check_reg_coords_negative(register_inst):
    reg_coords = np.ones((5, 2), dtype=np.float32)
    reg_coords[2, :] = [-.1, 92.1]
    register_inst.registered_coords = reg_coords
    reg_ok = register_inst.check_reg_coords()
    assert reg_ok is False


def test_check_reg_coords_already_false(register_inst):
    reg_coords = np.ones((5, 2), dtype=np.float32)
    register_inst.registered_coords = reg_coords
    register_inst.registration_ok = False
    reg_ok = register_inst.check_reg_coords()
    assert reg_ok is False
