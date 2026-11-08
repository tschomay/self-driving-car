"""Fast checks for the Phase 2 camera: orientation, edges, frame stacking, SB3 adapter."""

import numpy as np

from sdc.racer import camera, sim
from sdc.racer import track as trk


def test_view_shape_and_orientation():
    b = sim.RacerBatch(1, tracks=[0], start="line")
    b.reset()
    img = camera.Camera().batch_views(b)[0]
    assert img.shape == (camera.SIZE, camera.SIZE) and img.dtype == np.uint8
    r, c = camera.CAR_PIXEL
    assert img[r, c] == camera.ROAD                      # the car sits on the road
    # road is about trk.WIDTH wide around the car: edges a few pixels either side, grass beyond
    half = trk.WIDTH / 2 / camera.M_PER_PX
    assert img[r, int(c - half - 4)] == camera.GRASS and img[r, int(c + half + 4)] == camera.GRASS
    assert camera.EDGE in img[r, int(c - half - 2):int(c - half + 2)]
    # left in the image is the car's left: pixel column 0 is LEFT > 0
    assert camera.LEFT[0, 0] > 0 and camera.FWD[0, 0] > camera.FWD[-1, 0]


def test_view_turns_with_car():
    cam = camera.Camera()
    t = trk.generate(0)
    p, h = t.center[0], np.arctan2(t.tangent[0, 1], t.tangent[0, 0])
    a = cam.view(0, p, h)
    # turned 90° left, the road (which ran up the image) now runs across it
    bimg = cam.view(0, p, h + np.pi / 2)
    assert (a[:, camera.SIZE // 2] == camera.ROAD).mean() > 0.5
    assert (bimg[camera.CAR_PIXEL[0], :] == camera.ROAD).mean() > 0.1
    assert (bimg[:camera.CAR_PIXEL[0] - 8, camera.SIZE // 2] == camera.GRASS).mean() > 0.5


def test_camera_vec_env_frames_and_terminal_obs():
    from sdc.racer.camera_env import CameraVecEnv

    env = CameraVecEnv(2, n_frames=3, speed=False, tracks=[0, 1], max_steps=5)
    o = env.reset()
    assert set(o) == {"image"} and o["image"].shape == (2, 3, 64, 64)
    assert np.all(o["image"][:, 0] == o["image"][:, 2])        # stack starts filled with the first image
    for k in range(5):
        env.step_async(np.tile([0.0, 1.0], (2, 1)))
        o, r, done, infos = env.step_wait()
    assert done.all()                                           # time limit
    assert infos[0]["terminal_observation"]["image"].shape == (3, 64, 64)
    assert np.all(o["image"][:, 0] == o["image"][:, 2])        # new episode: stack restarted
    assert not np.array_equal(infos[0]["terminal_observation"]["image"][2], o["image"][0, 2])


def test_random_shift():
    rng = np.random.default_rng(0)
    imgs = np.arange(2 * 3 * 8 * 8, dtype=np.uint8).reshape(2, 3, 8, 8)
    assert camera.random_shift(imgs, 0, rng) is imgs
    out = camera.random_shift(imgs, 2, rng)
    assert out.shape == imgs.shape and out.dtype == imgs.dtype
    # each image is a translated copy: some 4x4 window of the original reappears
    for k in range(6):
        a, b = imgs.reshape(6, 8, 8)[k], out.reshape(6, 8, 8)[k]
        assert any(np.array_equal(a[2:6, 2:6], b[2 + dy:6 + dy, 2 + dx:6 + dx])
                   for dy in range(-2, 3) for dx in range(-2, 3))
