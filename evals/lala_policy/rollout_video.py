"""High-resolution rollout videos for the policy drivers (run_{robocasa,libero}_eval --video-res).

The rollout GIF the drivers always wrote is a sanity check: one camera, the policy's observation shrunk to 128 px,
every 3rd / 4th step. --video-res R adds <task>_ep<NN>_<ok|fail>.mp4 for the same episodes: every step, the
video cameras side by side, each rendered at R x R straight from the simulator (env.sim.render), H.264 at the
control frequency -- and the GIF of those episodes is then made from the R x R renders of the first video camera
(same name, same frame sampling, R px instead of 128). The policy still receives its own observation (224 / 256
px): the extra renders read the
simulator state and change neither it nor the observation rendered from it (checked on RoboCasa over moving
steps: qpos / qvel / ctrl / time and the three 224 px observations bit-identical before and after).

Cost: with this image's software renderer an extra render takes about as long at 768 px as at 224 px, ~0.13 s
per camera and step against ~0.43 s for a plain RoboCasa step -- three cameras roughly double an episode, one adds
about a third. R equal to the driver's --env-res reuses the observation and costs nothing.
Frames are streamed to ffmpeg, so an episode is never held in memory.
"""

import os
import shutil
import subprocess

import numpy as np
from PIL import Image


def add_arguments(ap, cameras):
    ap.add_argument(
        "--video-res",
        type=int,
        default=0,
        help="also write an MP4 of every episode that gets a GIF: every step, the --video-cams side by side, "
        "each rendered at this resolution from the simulator (0 = off; the policy's observation is untouched)",
    )
    ap.add_argument("--video-cams", nargs="+", default=list(cameras), help="simulator camera names for --video-res")


class RolloutVideo:
    """One episode. frame(env_sim, obs) once per step (and for the last observation), then close(mp4[, gif])."""

    def __init__(self, tmp_path, res, cameras, env_res, fps=20, flip=True, gif_every=4):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("--video-res needs ffmpeg on PATH")
        if int(res) % 2:
            raise ValueError(f"--video-res must be even (yuv420p), got {res}")
        self.tmp_path, self.res, self.cameras, self.env_res, self.flip = (
            tmp_path,
            int(res),
            list(cameras),
            env_res,
            flip,
        )
        size = f"{self.res * len(self.cameras)}x{self.res}"
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size]
        cmd += ["-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
        cmd += ["-movflags", "+faststart", "-f", "mp4", tmp_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.gif_every, self.gif_frames, self.n = int(gif_every), [], 0  # first camera, every gif_every-th step

    def frame(self, sim, obs):
        views = []
        for cam in self.cameras:
            if self.res == self.env_res and f"{cam}_image" in obs:  # the observation already is that render
                im = obs[f"{cam}_image"]
            else:
                im = sim.render(camera_name=cam, width=self.res, height=self.res, depth=False)
            views.append(np.flipud(im) if self.flip else im)
        if self.n % self.gif_every == 0:
            self.gif_frames.append(np.ascontiguousarray(views[0]))
        self.n += 1
        self.proc.stdin.write(np.ascontiguousarray(np.concatenate(views, axis=1), dtype=np.uint8).tobytes())

    def close(self, path, gif_path=None):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.tmp_path}")
        os.replace(self.tmp_path, path)
        if gif_path is not None and self.gif_frames:
            ims = [Image.fromarray(f) for f in self.gif_frames]
            ims[0].save(gif_path, save_all=True, append_images=ims[1:], duration=100, loop=0)
