# LaLa-World planning evaluation

MPC with CEM in latent space, following the JEPA-WMs and DINO-WM protocols. One planning step = one world-model
transition = one action chunk (5 simulator steps, or one push for rope / granular). CEM samples chunk sequences in
the normalized action space of `data_preparation/specs/<env>_v1.yaml`, renders each candidate with the training
text template, rolls the predictor out `--horizon` steps and scores the last latent against the encoded goal.
`--num-act-stepped` chunks are executed before replanning. `--ctxt-window` is the number of (frame, text) blocks
the predictor sees (3 for the paper's simulation runs, 2 for DROID / RoboCasa).

| benchmark | episodes | length | horizon | executed / plan | CEM iters x samples x elites | success |
|---|---|---|---|---|---|---|
| wall | 96 | 6 | 6 | 6 | 30 x 300 x 10 | final agent position error < 4.5 px |
| point_maze | 96 | 6 | 6 | 6 | 30 x 300 x 10 | final ball position error < 0.5 |
| pusht | 96 | 6 | 6 | 6 | 30 x 300 x 10 | pusher + T position error < 20 and angle error < 20 deg |
| metaworld reach / reach-wall | 48 | 20 | 6 | 3 | 15 x 300 x 10 | env success flag at any step |
| rope / granular | 24 / 12 | 5 | 5 | 1 | 30 x 300 x 30 | final Chamfer distance < 0.900 / 0.414 |
| droid (offline) | 64 | 3 | 3 | - | 15 x 300 x 10 | Action Score = max(0, 800 (0.1 - xyz error)) |
| robocasa reach / place | 32 | 12 | 3 | 1 | 15 x 300 x 10 | subtask condition at the final step |

```bash
export LALA_DATA_ROOT=/path/to/data; CK=<lala-world checkpoint>
python -m evals.lala_world.run_planning --ckpt $CK --env wall --episodes 96
python -m evals.lala_world.run_planning --ckpt $CK --env point_maze --episodes 96
python -m evals.lala_world.run_planning --ckpt $CK --env pusht --episodes 96
python -m evals.lala_world.run_planning --ckpt $CK --env metaworld --mw-task reach --goal-H 20 --horizon 6 --iterations 15 --num-act-stepped 3 --episodes 48
python -m evals.lala_world.run_planning --ckpt $CK --env rope --goal-H 5 --horizon 5 --iterations 30 --num-elites 30 --num-act-stepped 1 --episodes 24
python -m evals.lala_world.run_planning --ckpt $CK --env granular --goal-H 5 --horizon 5 --iterations 30 --num-elites 30 --num-act-stepped 1 --episodes 12
python -m evals.lala_world.run_planning_droid_franka --ckpt $CK --horizon 3 --ctxt-window 2 --episodes 64
bash evals/lala_world/driver_robocasa.sh $CK reach 0 plan 32      # RoboCasa venv; also run the floors: hold | zero | random
bash evals/lala_world/driver_robocasa.sh $CK place 0 plan 32
```

Results: `<out>/<env>/<tag>/result.json` (summary + per-episode rows), `ep*_init_goal_final.png`, `ep*.gif`
(`--out`, default `logs/planning`). The episode seed is `seed^2 + ep * seed`, so `--ep-offset N --episodes M` shards a
run over GPUs and the shards merge by episode index; keep the same split when rerunning (the CEM noise also depends on
the position inside a shard).

- Rope / granular need PyFleX (`PYFLEX_HOME`, see Installation); `EGL_GPU` selects the rendering GPU and must differ
  between shards. `--targets` pins the (init, goal) problem set; the seed-1 sets of the paper ship in `targets/`.
- Meta-World needs `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`; point_maze is a modern-MuJoCo port of the D4RL maze
  (`point_maze_env/`, parity-checked against the dataset).
- DROID and RoboCasa use the evaluation-only episodes released with JEPA-WMs under `$LALA_DATA_ROOT/eval_only/`
  (`franka_custom/`, `robocasa/combine_all_im256.hdf5`); RoboCasa runs zero-shot in the DROID action space.
- `probe_onestep.py` checks that one-step predictions beat copying the current frame before planning;
  `--save-pred-pca N` / `--save-pred-decode N --decoder-ckpt <ckpt>` save the imagined rollouts;
  `--wm rae` plans with the RAEv2 baseline (`RAEv2/`).
