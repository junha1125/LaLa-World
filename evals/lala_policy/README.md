# LaLa-Policy evaluation

Direct policy rollouts with the Cosmos Policy protocol. The policy receives the task instruction, the proprioceptive
state and the camera views, predicts a chunk of K actions, and `--open-loop` of them are executed before the next query.

| | LIBERO | RoboCasa |
|---|---|---|
| episodes | 4 suites x 10 tasks x 50 benchmark init states | 24 tasks x 50 (5 held-out scenes x 10, object split B) |
| step budget | 220 / 280 / 300 / 520 (spatial / object / goal / 10) | 300-1000 per task |
| chunk / executed | 16 / 16 | 32 / 16 |
| views | agentview + wrist | left + right + wrist |
| success | env `done` | `env._check_success()` |
| python | `third_party/venv-libero` (`$LIBERO_PYTHON`) | `third_party/venv-robocasa` (`$ROBOCASA_PYTHON`) |

```bash
CK=<lala-policy checkpoint>
bash evals/lala_policy/run_libero_all.sh $CK <tag> 50 16        # 40 processes, 16 at a time, round-robin over GPUs
bash evals/lala_policy/run_robocasa_all.sh $CK <tag> 50 12
python -m evals.lala_policy.collect --out logs/policy_eval      # summary.md / summary.json, Wilson 95% CIs
bash evals/lala_policy/driver_libero.sh $CK libero_spatial 0 2 smoke --tasks 0 --max-steps 40       # smoke test
bash evals/lala_policy/driver_robocasa.sh $CK 0 1 smoke --task PnPCounterToCab --max-steps 20
```

Results: `<out>/libero/<tag>/<suite>/task<k>.json`, `<out>/robocasa/<tag>/<task>.json` (`$LALA_POLICY_EVAL_OUT`,
default `logs/policy_eval`), written after every episode; finished tasks are skipped on rerun (`--overwrite`). The first
two episodes of each task are saved as GIFs. The simulators render with Mesa / EGL on the CPU, so run many processes.

- `--seed` is the sampling seed.
- `--save-imagine N --decoder-ckpt <decoder>` compares every query's imagined frame at t + K (policy pass and
  world-model pass on the sampled chunk) with the frame the simulator reaches; `imagine_viz.py --out <root>` aggregates.
- `--video-res 512` writes an MP4 of every step for the episodes that get a GIF.
