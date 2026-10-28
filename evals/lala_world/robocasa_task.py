"""`PnPCounterTop`: the custom RoboCasa task the JEPA-WM eval configs name.

The eval configs (`evals/simu_env_planning/base_configs/rcasa_custom/*.yaml`) ask for
`robocasa-PnPCounterTop`, and `evals/simu_env_planning/envs/robocasa.py` appends that name to the
task registry by hand -- it lives in the authors' private robocasa fork ("robocasa-murp" in the
teleop dataset's asset paths), not in public robocasa v0.2. This module re-declares it so
`robosuite.make("PnPCounterTop")` resolves: robosuite's `EnvMeta` metaclass registers every
`MujocoEnv` subclass at class-creation time, so importing this file is the registration.

Only the *dummy* env goes through this class' own placement logic. Every eval episode calls
`RoboCasaWrapper.prepare()`, which loads the demo's `ep_meta` (its own `object_cfgs` and
`fixture_refs` win over `_get_obj_cfgs`/`_setup_kitchen_references`) and then replaces the whole
scene via `reset_from_xml_string(model_file)`. So the class only has to (a) build standalone and
(b) expose `self.objects["obj"]` + `obj_body_id`, which `subtask_success()` reads. The teleop
`ep_meta` refs are named `wall` (a sink fixture) and `counter`; registering the same two names
keeps `_setup_kitchen_references`'s deserialization path a no-op re-lookup.

Success is left at the `Kitchen` default (always False): the eval scores the reach/pick/place
subtask conditions in `RoboCasaWrapper.subtask_success()`, never the task's own `_check_success`.
"""

from copy import deepcopy

from robocasa.environments.kitchen.kitchen import FixtureType
from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnP
from robocasa.utils.camera_utils import CAM_CONFIGS


def sanitize_ep_meta(ep_meta):
    """Strip the fork-specific scene wiring out of a demo's `ep_meta`.

    The teleop demos were recorded against the authors' robocasa fork, whose kitchen layouts carry
    fixtures public v0.2 does not build (layout 4 has 94 fixtures there vs 67 here, and names like
    `counter_1_left_group` simply do not exist). `_setup_kitchen_references` and the placement
    sampler resolve those names by string and assert when they miss.

    Nothing about that scene survives: `RoboCasaWrapper.prepare()` immediately calls
    `reset_from_xml_string(model_file)`, which rebuilds the sim from the demo's own recorded XML,
    and then loads the recorded qpos/qvel. The reset only has to construct without raising and
    leave `self.objects["obj"]` pointing at the right mesh -- which `object_cfgs[0]["info"]`
    pins independently of where the sampler would have put it. So we drop `fixture_refs` (our
    `_setup_kitchen_references` then resolves `wall`/`counter` by fixture type against whatever
    layout is available) and drop `placement` (the sampler skips objects that have none).
    """
    meta = deepcopy(ep_meta)
    meta.pop("fixture_refs", None)
    for cfg in meta.get("object_cfgs", []):
        cfg.pop("placement", None)
    return meta


def register_teleop_cameras(cam_configs):
    """Make the demos' cameras part of the robot's camera set.

    The teleop demos were filmed through `robot0_leftview`, a camera the paper re-aimed at the
    DROID left-camera viewpoint; public robocasa v0.2 has no such entry in `CAM_CONFIGS`, so a
    freshly built env would have no camera by that name and `Kitchen.edit_model_xml` (which
    rewrites every camera it knows about) would leave the demo XML's own copy alone. Registering
    the recorded `ep_meta["cam_configs"]` makes both paths agree on one pose.
    """
    for name, cfg in cam_configs.items():
        CAM_CONFIGS["DEFAULT"][name] = deepcopy(cfg)


class PnPCounterTop(PnP):
    """Pick a graspable object off the counter next to the sink and place it back on the counter."""

    _pruned_once = False

    def __init__(self, obj_groups="all", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # ref names must match the teleop ep_meta ("wall" -> sink_left_group, "counter" ->
        # counter_1_left_group); with ep_meta loaded both are already in self.fixture_refs and
        # register_fixture_ref just hands them back.
        try:
            self.wall = self.register_fixture_ref("wall", dict(id=FixtureType.SINK))
            self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.wall))
        except AssertionError:  # a layout without a sink: any counter will do for the scratch scene
            self.wall = None
            self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER))
        self.init_robot_base_pos = self.counter

    def reset_from_xml_string(self, xml_string):
        """Load the demo's recorded scene, dropping the parts of the old one it does not contain.

        Upstream this method swaps the *simulation* for the XML and then resets, leaving the
        python-side scene description (`self.fixtures`, `self.model.mujoco_objects`) describing
        whatever `_load_model` last built. That is fine when the two agree -- which they do for the
        authors, who recorded and replay with the same robocasa fork. Public v0.2 lays out its
        kitchens differently (layout 4: 67 fixtures here vs 94 in the demos), so the reset scene
        carries fixtures the demo XML never had, and the reset walks straight into them:
        `_setup_references` asks for their geom ids, `update_state` for their site positions.

        So prune between loading the sim and resetting on it. What survives is everything the
        eval actually reads -- the robot, the manipulated object, and the fixtures the demo does
        have -- because those are in both scenes by construction. `_load_model` rebuilds the full
        description on the next ordinary reset, so nothing accumulates across episodes.
        """
        self.close()
        self.deterministic_reset = True
        self._initialize_sim(xml_string=xml_string)
        self._prune_to_sim()
        self.reset()
        self.deterministic_reset = False

    def _prune_to_sim(self):
        geoms = set(self.sim.model.geom_names)
        sites = set(self.sim.model.site_names)

        def present(model):
            names = set(model.visual_geoms) | set(model.contact_geoms)
            return names <= geoms and set(model.sites) <= sites

        dropped = [name for name, fxtr in self.fixtures.items() if not present(fxtr)]
        if not dropped:
            return
        self.fixtures = {name: f for name, f in self.fixtures.items() if name not in dropped}
        self.model.mujoco_objects = [m for m in self.model.mujoco_objects if present(m)]
        self.fixture_refs = {k: v for k, v in self.fixture_refs.items() if v.name not in dropped}
        if not type(self)._pruned_once:
            type(self)._pruned_once = True
            print(
                f"[PnPCounterTop] demo scene lacks {len(dropped)} fixtures of the locally built "
                f"layout (e.g. {', '.join(dropped[:4])}) -- pruned; reported once"
            )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"pick the {self.get_obj_lang()} from the counter"
        return ep_meta

    def _get_obj_cfgs(self):
        """One graspable object on the counter -- the dummy scene only, never an eval episode.

        The recorded `object_cfgs` aim at a 10x10cm box in the counter's "left" reset region; that
        region is a property of the *fork's* layouts and comes back empty in several public v0.2
        ones, which the sampler turns into a hard `ValueError` rather than a retry. A plain
        front-of-counter placement builds in every layout, and the episode state is loaded from
        the demo anyway.
        """
        return [
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(fixture=self.counter, size=(0.30, 0.30), pos=(0.0, -1.0)),
            )
        ]
