import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg


class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()

        # # Get odor intensity
        # odor_gain = -500
        # odor_gradient = obs.odor_intensity[1] - obs.odor_intensity[0]
        # attractive_bias = odor_gain * odor_gradient/obs.odor_intensity.mean()
        # direction = int(attractive_bias > 0)

        # motor_gain = 1
        # control_signal[direction] = motor_gain*abs(attractive_bias)

        # # # Get vision image
        # fly_vision = obs.vision


    def get_actions(self, obs: Observation) -> Action:
        # Extract odor intensity from observations
        odor_intensity = obs.get("odor_intensity", None)
        if odor_intensity is not None:
            # Example logic: Move toward the direction with the highest odor intensity
            # print("Odor intensity:", odor_intensity.shape)
            # left_odor = odor_intensity[0].mean()  # Left antenna
            # right_odor = odor_intensity[1].mean()  # Right antenna
            attractive_intensities = np.average(
                obs['odor_intensity'][0,:].reshape(2,2), axis=0, weights=[9,1]
            )
            odor_gradient = attractive_intensities[0] - attractive_intensities[1]
            attractive_bias = -500 * odor_gradient / np.mean(odor_intensity)
            effective_bias = np.tanh(attractive_bias**2) * np.sign(attractive_bias)
            direction = int(effective_bias > 0)
            control_signal = np.ones(2)
            control_signal[direction] -= np.abs(effective_bias)
            action = np.array(control_signal)
            
        else:
            action = np.array([1.0, 1.0])  # Default forward action

        # Generate joint angles and adhesion using the CPG network
        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
