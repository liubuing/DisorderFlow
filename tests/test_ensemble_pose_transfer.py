import numpy as np

from ensemble_pose_transfer import kabsch_transform


def test_kabsch_transform_recovers_rigid_transform():
    moving = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = moving @ rotation + np.array([4.0, -2.0, 7.0])
    fitted_rotation, translation, rmsd = kabsch_transform(moving, target)
    assert rmsd < 1e-10
    assert np.allclose(moving @ fitted_rotation + translation, target)


def test_kabsch_transform_rejects_too_few_points():
    try:
        kabsch_transform(np.zeros((2, 3)), np.zeros((2, 3)))
    except ValueError as error:
        assert "three" in str(error)
    else:
        raise AssertionError("Expected a ValueError")
