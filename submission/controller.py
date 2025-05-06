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

    def get_odor_bias(self, obs: Observation):
        odor_intensity = obs.get("odor_intensity", None)
        if odor_intensity is not None:
            # Example logic: Move toward the direction with the highest odor intensity
            attractive_intensities = np.average(
                obs['odor_intensity'][0,:].reshape(2,2), axis=0, weights=[9,1]
            )
            odor_gradient = attractive_intensities[0] - attractive_intensities[1]
            attractive_bias = -500 * odor_gradient / np.mean(attractive_intensities)
            attractive_bias = attractive_bias/10
            effective_bias = np.tanh(attractive_bias**2) * np.sign(attractive_bias)
            direction = int(effective_bias > 0)
            odor_bias = np.zeros(2)
            odor_bias[direction] = np.abs(effective_bias)

            # vision_updated = obs.get("vision_updated", False)
            # if vision_updated:
            #     print("Odor bias: ", attractive_bias)
            
        else:
            odor_bias = np.zeros(2)  

        return odor_bias
    
    def alt_sigmoid(x):
        return 1 / (2*(1 + np.exp(-(x-6))))
    
    def get_vision_bias(self, obs: Observation):
        
        vision = obs.get("vision", None)
        if vision is not None:
            fly_vision = obs["vision"]
            left_eye = fly_vision[0,:,:]
            right_eye = fly_vision[1,:,:]

            # Yellow : dark
            yellow_left = left_eye[:,0].mean()
            yellow_right = right_eye[:,0].mean()
            
            # Pale : light
            pale_left = left_eye[:,1].mean()
            pale_right = right_eye[:,1].mean()

            yellow_gradient = yellow_left - yellow_right
            pale_gradient = pale_left - pale_right

            # vision_updated = obs.get("vision_updated", False)
            # if vision_updated:
            #     print("yellow gradient", np.round(yellow_gradient,4))
            #     print("pale gradient  ", np.round(pale_gradient,4))

            repulsive_bias = -150 * yellow_gradient - 100 * pale_gradient

            # effective_bias = np.tanh((repulsive_bias-attractive_bias)**2) * np.sign(repulsive_bias-attractive_bias)
            bias = repulsive_bias
            bias = bias/2 # 3
            effective_bias = np.tanh(bias**2) * np.sign(bias)

            direction = int(effective_bias > 0)
            vision_bias = np.zeros(2)
            vision_bias[direction] = np.abs(effective_bias)

            # vision_updated = obs.get("vision_updated", False)
            # if vision_updated:
            #     print("Vision bias: ", bias)

            return vision_bias
                
        return np.zeros(2)  # Default bias if no vision data is available
    
    def get_threat_response(self, obs: Observation):
        
        vision = obs.get("vision", None)
        if vision is not None:
            fly_vision = obs["vision"]
            left_eye = fly_vision[0,:,:]
            right_eye = fly_vision[1,:,:]

            # Yellow : dark
            yellow_left = left_eye[:,0].mean()
            yellow_right = right_eye[:,0].mean()
            # yellow intensity is more affected by red light
            # mean of yellow intensity when no threat : 0.68
            # mean of yellow intensity when threat : 0.66
            
            # Pale : light
            pale_left = left_eye[:,1].mean()
            pale_right = right_eye[:,1].mean()
            # mean of pale intensity when no threat : 0.29
            # mean of pale intensity when threat : 0.27

            # vision_updated = obs.get("vision_updated", False)
            # if vision_updated:
            #     print("Yellow left: ", yellow_left)
            #     print("Pale left: ", pale_left)
            #     print("Yellow right: ", yellow_right)
            #     print("Pale right: ", pale_right)

            threat_left = False
            threat_right = False
            threat = False

            if (pale_left <= 0.27) or (yellow_left <= 0.66):
                threat_left = True
                threat = True

            if (pale_right <= 0.27) or (yellow_right <= 0.66):
                threat_right = True
                threat = True

            # yellow_gradient = yellow_left - yellow_right
            # pale_gradient = pale_left - pale_right

            # repulsive_bias = -0 * yellow_gradient
            # attractive_bias = 200 * pale_gradient

            # Threat response -> speed up
            # if np.abs(repulsive_bias-attractive_bias) > 5:

            #     vision_updated = obs.get("vision_updated", False)
            #     if vision_updated:
            #         print("Threat detected! : ", repulsive_bias-attractive_bias)

            #     effective_bias = np.tanh((repulsive_bias-attractive_bias)**2) * np.sign(repulsive_bias-attractive_bias)
            #     direction = int(effective_bias < 0)
            #     threat_response = np.zeros(2)
            #     threat_response[direction] = np.abs(effective_bias)

            #     return threat_response
            
            action = np.zeros(2)

            # if threat:
            #     action = np.ones(2)
        
            if threat_left:
                vision_updated = obs.get("vision_updated", False)
                if vision_updated:
                    print("Left : ", threat_left)
                    print("Yellow left: ", yellow_left)
                    print("Pale left: ", pale_left)

                action[0] = 0.75
            
            if threat_right:
                vision_updated = obs.get("vision_updated", False)
                if vision_updated:
                    print("Right : ", threat_right)
                    print("Yellow right: ", yellow_right)
                    print("Pale right: ", pale_right)

                action[1] = 0.75
                
        return action


    def get_actions(self, obs: Observation) -> Action:
        
        action = np.ones(2)
        
        # Extract odor intensity from observations
        odor_bias = self.get_odor_bias(obs)
        # action -= odor_bias

        # Extract vision information from observations
        vision_bias = self.get_vision_bias(obs)
        # action -= vision_bias
        
        # global_bias = 0.55*odor_bias + 0.45*vision_bias
        global_bias = np.tanh(0.25 * np.arctanh(np.clip(odor_bias,0,0.999)) + 0.75 * np.arctanh(np.clip(vision_bias,0,0.999)))
        # vision_updated = obs.get("vision_updated", False)
        # if vision_updated:
        #     print("Global bias: ", global_bias)
        action -= global_bias

        # Threat response
        threat_response = self.get_threat_response(obs)

        for i in range(len(action)):
            action[i] = np.tanh((threat_response[i]) + np.arctanh(np.clip(action[i],0,0.999)))

        # np.clip(action, np.zeros(2), np.ones(2), out=action)

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
