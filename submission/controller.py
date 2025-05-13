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
        self.odor_memory = []
        self.vision_memory_pos = []
        self.vision_memory_vel = []
        self.vision_memory_acc = []

    def get_odor_bias(self, obs: Observation):
        odor_intensity = obs.get("odor_intensity", None)
        if odor_intensity is not None:
            # Example logic: Move toward the direction with the highest odor intensity
            attractive_intensities = np.average(
                obs['odor_intensity'][0,:].reshape(2,2), axis=0, weights=[9,1]
            )
            # odor_gradient = attractive_intensities[0] - attractive_intensities[1]
            # attractive_bias = -500 * odor_gradient / np.mean(attractive_intensities)
            # attractive_bias = attractive_bias/10
            # effective_bias = np.tanh(attractive_bias**2) * np.sign(attractive_bias)
            # direction = int(effective_bias > 0)
            # odor_bias = np.zeros(2)
            # odor_bias[direction] = np.abs(effective_bias)

            vision_updated = obs.get("vision_updated", False)
            # if vision_updated:
            #     print("Odor bias: ", attractive_bias)

            left_attr_odor = attractive_intensities[0] 
            right_attr_odor = attractive_intensities[1]

            mean_attr_odor = np.mean(attractive_intensities)

            odor_gradient = left_attr_odor-right_attr_odor

            turning_bias = 50*odor_gradient/mean_attr_odor

            self.odor_memory.append([left_attr_odor, right_attr_odor, odor_gradient]) 

            if len(self.odor_memory) > 10:
                self.odor_memory.pop(0)

            if len(self.odor_memory) > 5: # Taking memories into consideration
                left_attr_odor_1 = np.mean(self.odor_memory[-2][0])
                right_attr_odor_1 = np.mean(self.odor_memory[-2][1])
                odor_gradient_1 = np.mean(self.odor_memory[-2][2])
                odor_gradient_std = np.std(self.odor_memory[0:-2][2])

                # Choose for each case 
                if left_attr_odor > right_attr_odor:
                    # Turn left
                    if (np.abs(odor_gradient) < 1e-5) or (turning_bias/mean_attr_odor < 100) : # Small gradient or big odor source --> turn less
                        # if vision_updated:
                        #     print(f"\nsoft turn : {odor_gradient}")
                        #     print(f"Bias {turning_bias}")
                        #     print(f"{np.abs(odor_gradient)-np.abs(odor_gradient_1)}")
                        # Gentle turn
                        r_speed = np.tanh(np.abs(turning_bias))
                        return np.array([1-r_speed, 1])
                    else: 
                        # if vision_updated:
                        #     print(f"\nhard turn : {odor_gradient}")
                        #     print(f"Bias {turning_bias}")
                        #     print(f"{np.abs(odor_gradient)-np.abs(odor_gradient_1)}")
                        # Hard turn
                        return np.array([-0.75, 1])

                else:
                    # Turn right
                    if  (np.abs(odor_gradient) < 1e-5) or (np.abs(turning_bias)/mean_attr_odor < 100) : # Small gradient or big odor source --> turn less
                        # if vision_updated:
                        #     print(f"\nsoft turn : {odor_gradient}")
                        #     print(f"Bias {turning_bias}")
                        #     print(f"{np.abs(odor_gradient)-np.abs(odor_gradient_1)}")
                        # Gentle turn
                        l_speed = np.tanh(np.abs(turning_bias))
                        return np.array([1, 1-l_speed])
                    else: # Angle >= 90°
                        # if vision_updated:
                        #     print(f"\nhard turn : {odor_gradient}")
                        #     print(f"Bias {turning_bias}")
                        #     print(f"{np.abs(odor_gradient)-np.abs(odor_gradient_1)}")
                        # Hard turn
                        return np.array([1, -0.75])
            
            else: # Not enough memories
                if left_attr_odor > right_attr_odor:
                    # Turn left
                    r_speed = np.tanh(np.abs(turning_bias))
                    return np.array([1-r_speed, 1])
                else:
                    # Turn right
                    l_speed = np.tanh(np.abs(turning_bias))
                    return np.array([1, 1-l_speed])



            
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

            # Mean pixels value --- Yellow : dark | Pale : light
            yellow_left = left_eye[:,0].mean()
            yellow_right = right_eye[:,0].mean()
            pale_left = left_eye[:,1].mean()
            pale_right = right_eye[:,1].mean()
            self.vision_memory_pos.append([yellow_left, pale_left, yellow_right, pale_right]) 
            if len(self.vision_memory_pos) > 4:
                self.vision_memory_pos.pop(0)

            dt = 0.0005 # 5s/10000it

            # Mean pixels changes (first derivative)
            if len(self.vision_memory_pos) >= 2:
                yellow_left_1 = self.vision_memory_pos[-2][0]
                pale_left_1 = self.vision_memory_pos[-2][1]
                yellow_right_1 = self.vision_memory_pos[-2][2]
                pale_right_1 = self.vision_memory_pos[-2][3]
                # First derivatives (velocity)
                yellow_left_velocity = (yellow_left - yellow_left_1) / dt
                pale_left_velocity = (pale_left - pale_left_1) / dt
                yellow_right_velocity = (yellow_right - yellow_right_1) / dt
                pale_right_velocity = (pale_right - pale_right_1) / dt
                self.vision_memory_vel.append([yellow_left_velocity, pale_left_velocity, yellow_right_velocity, pale_right_velocity])
            if len(self.vision_memory_vel) > 4:
                self.vision_memory_vel.pop(0)
            
            # Mean pixels looms (second derivative)
            if len(self.vision_memory_vel) >= 2:
                yellow_left_velocity_1 = self.vision_memory_vel[-2][0]
                pale_left_velocity_1 = self.vision_memory_vel[-2][1]
                yellow_right_velocity_1 = self.vision_memory_vel[-2][2]
                pale_right_velocity_1 = self.vision_memory_vel[-2][3]
                # Second derivatives (acceleration)
                yellow_left_acceleration = (yellow_left_velocity - yellow_left_velocity_1) / dt
                pale_left_acceleration = (pale_left_velocity - pale_left_velocity_1) / dt
                yellow_right_acceleration = (yellow_right_velocity - yellow_right_velocity_1) / dt
                pale_right_acceleration = (pale_right_velocity - pale_right_velocity_1) / dt
                self.vision_memory_acc.append([yellow_left_acceleration, pale_left_acceleration, yellow_right_acceleration, pale_right_acceleration])
            if len(self.vision_memory_acc) > 3:
                self.vision_memory_acc.pop(0)

            # Normal vision response
            yellow_gradient = yellow_left - yellow_right
            pale_gradient = pale_left - pale_right

            bias = -50 * yellow_gradient - 50 * pale_gradient
            effective_bias = np.tanh(bias) # if left obstacle -> effective bias > 0
            obstacle_pos = int(effective_bias < 0) # if left obstacle -> position = 0 -> reduce right side velocity

            vision_response = np.ones(2)
            for i in range(len(vision_response)):
                vision_response[i] = 1 - np.abs(effective_bias)*np.abs(i-obstacle_pos)

            # Vision change response
            # if approaching on right -> yellow/pale_right_velocity < 0 -> decrease left speed
            if len(self.vision_memory_pos) < 2:
                yellow_left_velocity = 0
                pale_left_velocity = 0
                yellow_right_velocity = 0
                pale_right_velocity = 0
            left_eye_change = (yellow_left_velocity + pale_left_velocity)/2
            right_eye_change = (yellow_right_velocity + pale_right_velocity)/2

            if left_eye_change < -2:
                vision_response[1] = np.clip(vision_response[1] - np.abs(left_eye_change)/20, -1, 1)
            if right_eye_change < -2:
                vision_response[0] = np.clip(vision_response[0] - np.abs(right_eye_change)/20, -1, 1)
                right_eye_change = 0

            
            
            
            vision_updated = obs.get("vision_updated", False)
            # if vision_updated:
            #     # print(f"Vision response: {vision_response}")
            #     # print(f"Effective bias : {effective_bias}")
            #     # print(f"Yellow gradient : {yellow_gradient}")
            #     # print(f"Pale gradient : {pale_gradient}")
            #     print(f"Left change {left_eye_change}")
            #     print(f"Right change {right_eye_change}")

            return vision_response


    def get_threat_response(self, obs: Observation):
        threat_threshold = -10000
        action = np.array([0, 0])

        if len(self.vision_memory_acc) > 2:
            # print(f"Length {len(self.vision_memory_acc)}")
            # print(f"Shape {np.shape(self.vision_memory_acc)}")
            # print(self.vision_memory_acc[-4:-1])
            # print(self.vision_memory_acc[0][:])
            # print([vec[0] for vec in self.vision_memory_acc])
            # print(self.vision_memory_acc[:][0])
            left_looming = np.mean([[vec[0] for vec in self.vision_memory_acc],[vec[1] for vec in self.vision_memory_acc]])
            right_looming = np.mean([[vec[2] for vec in self.vision_memory_acc],[vec[3] for vec in self.vision_memory_acc]])

            vision_updated = obs.get("vision_updated", False)
            # if vision_updated:
            #     print(f"Left looming {left_looming}")
            #     print(f"Right looming {right_looming}")

            if left_looming > right_looming: # Priority on left threat
                if left_looming < threat_threshold:
                    # Fast going back for right legs
                    action = np.array([-1, -1])

            else: # Priority on right threat
                if right_looming < threat_threshold:
                    # Fast going back for left legs
                    action = np.array([-1, -1])
        
        return action

    # possible implementation of heading correction to filter the artifacts caused by head movements
    
    # def get_vision_bias(self, obs: Observation):
        
    #     vision = obs.get("vision", None)
    #     if vision is not None:
    #         fly_vision = obs["vision"]
    #         left_eye = fly_vision[0,:,:]
    #         right_eye = fly_vision[1,:,:]

    #         # Yellow : dark
    #         yellow_left = left_eye[:,0].mean()
    #         yellow_right = right_eye[:,0].mean()
            
    #         # Pale : light
    #         pale_left = left_eye[:,1].mean()
    #         pale_right = right_eye[:,1].mean()

    #         yellow_gradient = yellow_left - yellow_right
    #         pale_gradient = pale_left - pale_right

    #         # vision_updated = obs.get("vision_updated", False)
    #         # if vision_updated:
    #         #     print("yellow gradient", np.round(yellow_gradient,4))
    #         #     print("pale gradient  ", np.round(pale_gradient,4))

    #         repulsive_bias = -150 * yellow_gradient - 100 * pale_gradient

    #         # effective_bias = np.tanh((repulsive_bias-attractive_bias)**2) * np.sign(repulsive_bias-attractive_bias)
    #         bias = repulsive_bias
    #         bias = bias/2 # 3
    #         effective_bias = np.tanh(bias**2) * np.sign(bias)

    #         direction = int(effective_bias > 0)
    #         vision_bias = np.zeros(2)
    #         vision_bias[direction] = np.abs(effective_bias)

    #         # vision_updated = obs.get("vision_updated", False)
    #         # if vision_updated:
    #         #     print("Vision bias: ", bias)

    #         return vision_bias
                
    #     return np.zeros(2)  # Default bias if no vision data is available
    
    # def get_threat_response(self, obs: Observation):
        
    #     vision = obs.get("vision", None)
    #     if vision is not None:
    #         fly_vision = obs["vision"]
    #         left_eye = fly_vision[0,:,:]
    #         right_eye = fly_vision[1,:,:]

    #         # Yellow : dark
    #         yellow_left = left_eye[:,0].mean()
    #         yellow_right = right_eye[:,0].mean()
    #         # yellow intensity is more affected by red light
    #         # mean of yellow intensity when no threat : 0.68
    #         # mean of yellow intensity when threat : 0.66
            
    #         # Pale : light
    #         pale_left = left_eye[:,1].mean()
    #         pale_right = right_eye[:,1].mean()
    #         # mean of pale intensity when no threat : 0.29
    #         # mean of pale intensity when threat : 0.27

    #         # vision_updated = obs.get("vision_updated", False)
    #         # if vision_updated:
    #         #     print("Yellow left: ", yellow_left)
    #         #     print("Pale left: ", pale_left)
    #         #     print("Yellow right: ", yellow_right)
    #         #     print("Pale right: ", pale_right)

    #         dt = 0.0005

    #         self.vision_memory_pos.append([yellow_left, pale_left, yellow_right, pale_right]) 

    #         if len(self.vision_memory_pos) > 4:
    #             self.vision_memory_pos.pop(0)

    #         if len(self.vision_memory_pos) >= 2:
    #             yellow_left_1 = self.vision_memory_pos[-2][0]
    #             pale_left_1 = self.vision_memory_pos[-2][1]
    #             yellow_right_1 = self.vision_memory_pos[-2][2]
    #             pale_right_1 = self.vision_memory_pos[-2][3]

    #             # First derivatives (velocity)
    #             yellow_left_velocity = (yellow_left - yellow_left_1) / dt
    #             pale_left_velocity = (pale_left - pale_left_1) / dt
    #             yellow_right_velocity = (yellow_right - yellow_right_1) / dt
    #             pale_right_velocity = (pale_right - pale_right_1) / dt

    #             self.vision_memory_vel.append([yellow_left_velocity, pale_left_velocity, yellow_right_velocity, pale_right_velocity])
            
    #         if len(self.vision_memory_vel) > 4:
    #             self.vision_memory_vel.pop(0)
            
    #         if len(self.vision_memory_vel) >= 2:
    #             yellow_left_velocity_1 = self.vision_memory_vel[-2][0]
    #             pale_left_velocity_1 = self.vision_memory_vel[-2][1]
    #             yellow_right_velocity_1 = self.vision_memory_vel[-2][2]
    #             pale_right_velocity_1 = self.vision_memory_vel[-2][3]

    #             # Second derivatives (acceleration)
    #             yellow_left_acceleration = (yellow_left_velocity - yellow_left_velocity_1) / dt
    #             pale_left_acceleration = (pale_left_velocity - pale_left_velocity_1) / dt
    #             yellow_right_acceleration = (yellow_right_velocity - yellow_right_velocity_1) / dt
    #             pale_right_acceleration = (pale_right_velocity - pale_right_velocity_1) / dt

    #             vision_updated = obs.get("vision_updated", False)
    #             if vision_updated:
    #                 # print("Yellow left acceleration: ", yellow_left_acceleration)
    #                 # print("Pale left acceleration: ", pale_left_acceleration)
    #                 # print("Yellow right acceleration: ", yellow_right_acceleration)
    #                 # print("Pale right acceleration: ", pale_right_acceleration)

    #                 print("Yellow left velocity: ", yellow_left_velocity)
    #                 print("Pale left velocity: ", pale_left_velocity)
    #                 print("Yellow right velocity: ", yellow_right_velocity)
    #                 print("Pale right velocity: ", pale_right_velocity)

    #             return np.zeros(2)
            

    #         return np.zeros(2)

    #     #     threat_left = False
    #     #     threat_right = False
    #     #     threat = False

    #     #     if (pale_left <= 0.27) or (yellow_left <= 0.66):
    #     #         threat_left = True
    #     #         threat = True

    #     #     if (pale_right <= 0.27) or (yellow_right <= 0.66):
    #     #         threat_right = True
    #     #         threat = True

    #     #     # yellow_gradient = yellow_left - yellow_right
    #     #     # pale_gradient = pale_left - pale_right

    #     #     # repulsive_bias = -0 * yellow_gradient
    #     #     # attractive_bias = 200 * pale_gradient

    #     #     # Threat response -> speed up
    #     #     # if np.abs(repulsive_bias-attractive_bias) > 5:

    #     #     #     vision_updated = obs.get("vision_updated", False)
    #     #     #     if vision_updated:
    #     #     #         print("Threat detected! : ", repulsive_bias-attractive_bias)

    #     #     #     effective_bias = np.tanh((repulsive_bias-attractive_bias)**2) * np.sign(repulsive_bias-attractive_bias)
    #     #     #     direction = int(effective_bias < 0)
    #     #     #     threat_response = np.zeros(2)
    #     #     #     threat_response[direction] = np.abs(effective_bias)

    #     #     #     return threat_response
            
    #     #     action = np.zeros(2)

    #     #     # if threat:
    #     #     #     action = np.ones(2)
        
    #     #     if threat_left:
    #     #         vision_updated = obs.get("vision_updated", False)
    #     #         if vision_updated:
    #     #             print("Left : ", threat_left)
    #     #             print("Yellow left: ", yellow_left)
    #     #             print("Pale left: ", pale_left)

    #     #         action[0] = 0.75
            
    #     #     if threat_right:
    #     #         vision_updated = obs.get("vision_updated", False)
    #     #         if vision_updated:
    #     #             print("Right : ", threat_right)
    #     #             print("Yellow right: ", yellow_right)
    #     #             print("Pale right: ", pale_right)

    #     #         action[1] = 0.75
                
    #     # return action


    # def get_actions(self, obs: Observation) -> Action:
        
    #     action = np.ones(2)
        
    #     # # Extract odor intensity from observations
    #     # odor_bias = self.get_odor_bias(obs)
    #     # # action -= odor_bias

    #     # # Extract vision information from observations
    #     # vision_bias = self.get_vision_bias(obs)
    #     # # action -= vision_bias
        
    #     # # global_bias = 0.55*odor_bias + 0.45*vision_bias
    #     # global_bias = np.tanh(0.25 * np.arctanh(np.clip(odor_bias,0,0.999)) + 0.75 * np.arctanh(np.clip(vision_bias,0,0.999)))
    #     # # vision_updated = obs.get("vision_updated", False)
    #     # # if vision_updated:
    #     # #     print("Global bias: ", global_bias)
    #     # action -= global_bias

    #     # # Threat response
    #     # threat_response = self.get_threat_response(obs)

    #     # for i in range(len(action)):
    #     #     action[i] = np.tanh((threat_response[i]) + np.arctanh(np.clip(action[i],0,0.999)))

    #     # np.clip(action, np.zeros(2), np.ones(2), out=action)

    #     action = self.get_odor_bias(obs)

    #     # Generate joint angles and adhesion using the CPG network
    #     joint_angles, adhesion = step_cpg(
    #         cpg_network=self.cpg_network,
    #         preprogrammed_steps=self.preprogrammed_steps,
    #         action=action,
    #     )

    #     return {
    #         "joints": joint_angles,
    #         "adhesion": adhesion,
    #     }

    def get_actions(self, obs: Observation):
        vision_updated = obs.get("vision_updated", False)

        odor_action = self.get_odor_bias(obs)
        vision_action = self.get_vision_bias(obs)
        threat_action = self.get_threat_response(obs)

        vision_power = np.tanh(np.abs(vision_action[0]-vision_action[1])) # when near 0, rely uniquely on odor, when increasing, increase vision 

        if (threat_action != 0).all():
            action = threat_action
            
            if vision_updated:
                print("Threat detected")

        # elif (vision_action.mean() > 0.9): # No obstacle
        #     action = 0.7*odor_action + 0.3*vision_action

        #     if vision_updated:
        #         print("No obstacle")

        else:
            action = (1-vision_power)*odor_action + vision_power*vision_action
            # action = 0.3*odor_action + 0.7*vision_action

        if vision_updated:
                print(f"Vision action {vision_action}")

        # action = 0.5*odor_action + 0.5*vision_action

        action = np.clip(action, -1, 1)

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
