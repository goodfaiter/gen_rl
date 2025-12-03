from pathlib import Path
import mujoco
import os
from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg

CARTPOLE_XML: Path = Path(os.path.dirname(__file__)) / "xmls" / "cartpole.xml"
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


def get_cartpole_robot_cfg() -> EntityCfg:
    """Get a fresh CartPole robot configuration instance."""
    return EntityCfg(
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    joint_names_expr=("slider",),
                    stiffness=100.0,
                    damping=10.0,
                    effort_limit=100.0,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = Entity(get_cartpole_robot_cfg())
    viewer.launch(robot.spec.compile())
