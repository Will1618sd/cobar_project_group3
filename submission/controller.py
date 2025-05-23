import numpy as np
import matplotlib.pyplot as plt
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
        self.odor_memory_pos = []
        self.odor_memory_vel = []
        self.vision_memory_pos = []
        self.vision_memory_vel = []
        self.vision_memory_acc = []

        self.position = np.zeros(2)
        self.heading  = 0.0 
        self.dt       = timestep 
        self.history = [self.position.copy()]
        self.vel_buffer = []
        self.buffer_size = 500
        self.heading_buffer = []
        self.heading_buffer_size = 500
        self.alpha = 0.7

        # Homing behavior
        self.home_position = self.position.copy()
        self.returning_home = False

        self.stuck_counter = 0
        self.stuck_threshold = 1000  # Number of steps to consider as stuck
        self.recovery_steps = 0
    
    @staticmethod
    def integrate_position(prev_pos: np.ndarray, v_rel: np.ndarray, heading: float, dt: float) -> np.ndarray:
        rot_back = np.array([
            [np.cos(heading),  -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])
        v_world = rot_back @ v_rel
        return prev_pos + v_world * dt
    
    def filter_velocity(self, obs: Observation) -> np.ndarray:
        # Low-pass filter on the velocity
        v_rel = obs.get("velocity", np.zeros(2))
        self.vel_buffer.append(v_rel)
        if len(self.vel_buffer) > self.buffer_size:
            self.vel_buffer.pop(0)
        return np.mean(self.vel_buffer, axis=0)

    def filter_heading(self, obs: Observation) -> float:
        # Low-pass filter on the heading
        raw_h = obs.get("heading", self.heading)
        self.heading_buffer.append(raw_h)
        if len(self.heading_buffer) > self.heading_buffer_size:
            self.heading_buffer.pop(0)
        angles = np.array(self.heading_buffer)
        sin_avg = np.sin(angles).mean()
        cos_avg = np.cos(angles).mean()
        return np.arctan2(sin_avg, cos_avg)

    def update_position_filters(self, v_filt: np.ndarray, h_filt: float):
        # Update the position and heading based on the filters
        self.position = self.integrate_position(self.position, v_filt, h_filt, self.dt)
        self.heading  = h_filt

    def get_odor_bias(self, obs: Observation):
        odor_intensity = obs.get("odor_intensity", None)
        if odor_intensity is not None:
            # Example logic: Move toward the direction with the highest odor intensity
            attractive_intensities = np.average(
                obs['odor_intensity'][0,:].reshape(2,2), axis=0, weights=[9,1]
            )

            odor_left, odor_right = attractive_intensities
            odor_intensity = np.mean(attractive_intensities)

            odor_speed = 1
            odor_gradient = odor_left-odor_right
            turning_bias = 50*odor_gradient/odor_intensity # turn left if turning_bias > 0, turn right if turning_bias < 0

            # Olfactive variation
            if len(self.odor_memory_pos) >= 2: # Taking memories into consideration
                odor_left_1 = self.odor_memory_pos[-1][0]
                odor_right_1 = self.odor_memory_pos[-1][1]
                odor_gradient_1 = self.odor_memory_pos[-1][2]
                odor_intensity_1 = self.odor_memory_pos[-1][3]

                odor_left_velocity = (odor_left - odor_left_1)/self.dt
                odor_right_velocity = (odor_right - odor_right_1)/self.dt
                odor_gradient_velocity = (odor_gradient - odor_gradient_1)/self.dt
                odor_intensity_velocity = (odor_intensity - odor_intensity_1)/self.dt

            if len(self.odor_memory_vel) >= 1: 
                # Filtering
                odor_left_velocity_1 = self.odor_memory_vel[-1][0]
                odor_right_velocity_1 = self.odor_memory_vel[-1][1]
                odor_gradient_velocity_1 = self.odor_memory_vel[-1][2]
                odor_intensity_velocity_1 = self.odor_memory_vel[-1][3]
                
                odor_left_velocity = self.alpha*odor_left_velocity + (1-self.alpha)*odor_left_velocity_1
                odor_right_velocity = self.alpha*odor_right_velocity + (1-self.alpha)*odor_right_velocity_1
                odor_gradient_velocity = self.alpha*odor_gradient_velocity + (1-self.alpha)*odor_gradient_velocity_1
                odor_intensity_velocity = self.alpha*odor_intensity_velocity + (1-self.alpha)*odor_intensity_velocity_1

                if ((np.abs(odor_gradient) > 1e-5) or odor_intensity_velocity < 0) and (turning_bias/odor_intensity > 100):
                    action = [0.55, 0.9*np.tanh(turning_bias*100)]

                else:
                    action = [odor_speed, turning_bias]
            
            else: # Not enough memories
                odor_speed = 0.5
                action = [odor_speed, turning_bias]

            if len(self.odor_memory_pos) >= 2:
                if odor_intensity_velocity < 0:
                    action = [0.55, 0.9*np.tanh(turning_bias*100)]
            

            # Olfactive memory
            self.odor_memory_pos.append([odor_left, odor_right, odor_gradient, odor_intensity]) 
            if len(self.odor_memory_pos) > 3:
                self.odor_memory_pos.pop(0)

            if len(self.odor_memory_pos) >= 3:
                self.odor_memory_vel.append([odor_left_velocity, odor_right_velocity, odor_gradient_velocity, odor_intensity_velocity]) 
                if len(self.odor_memory_vel) > 3:
                    self.odor_memory_vel.pop(0)

            return action
            
        else: 
            return [0, 0]  
 

    def get_vision_bias(self, obs: Observation):
        
        vision = obs.get("vision", None)
        if vision is not None:
            fly_vision = obs["vision"] # pixels have range of [0, 1], no obstacle : mean ~ 0.5
            left_eye = fly_vision[0,:,:]
            right_eye = fly_vision[1,:,:]

            # Mean pixels value --- Yellow : dark | Pale : light

            # Yellow : few obstacle ~ 0.70, large obstacle ~ 0.40
            yellow_left = left_eye[:,0].mean()
            yellow_right = right_eye[:,0].mean()

            # Pale : few obstacle ~ 0.30, large obstacle ~ 0.23
            pale_left = left_eye[:,1].mean()
            pale_right = right_eye[:,1].mean()

            # Dark pixels 
            dark_left = np.sum(left_eye[:, 0] == 0)
            dark_right = np.sum(right_eye[:, 0] == 0)

            # Mean pixels changes (first derivative)
            if len(self.vision_memory_pos) >= 1:
                yellow_left_1 = self.vision_memory_pos[-1][0]
                pale_left_1 = self.vision_memory_pos[-1][1]
                yellow_right_1 = self.vision_memory_pos[-1][2]
                pale_right_1 = self.vision_memory_pos[-1][3]

                dark_left_1 = self.vision_memory_pos[-1][4]
                dark_right_1 = self.vision_memory_pos[-1][5]
                # First derivatives (velocity)
                yellow_left_velocity = round((yellow_left - yellow_left_1) / self.dt, 1)
                pale_left_velocity = round((pale_left - pale_left_1) / self.dt, 1)
                yellow_right_velocity = round((yellow_right - yellow_right_1) / self.dt, 1)
                pale_right_velocity = round((pale_right - pale_right_1) / self.dt, 1)

                dark_left_velocity = round((dark_left - dark_left_1) / self.dt)
                dark_right_velocity = round((dark_right - dark_right_1) / self.dt)

            
            # Mean pixels looming (second derivative)
            if len(self.vision_memory_vel) >= 1: 
                yellow_left_velocity_1 = self.vision_memory_vel[-1][0]
                pale_left_velocity_1 = self.vision_memory_vel[-1][1]
                yellow_right_velocity_1 = self.vision_memory_vel[-1][2]
                pale_right_velocity_1 = self.vision_memory_vel[-1][3]

                dark_left_velocity_1 = self.vision_memory_vel[-1][4]
                dark_right_velocity_1 = self.vision_memory_vel[-1][5]

                yellow_left_velocity = round(yellow_left_velocity, 2)
                pale_left_velocity = round(pale_left_velocity, 2)
                yellow_right_velocity = round(yellow_right_velocity, 2)
                pale_right_velocity = round(pale_right_velocity, 2)

                dark_left_velocity = round(dark_left_velocity)
                dark_right_velocity = round(dark_right_velocity)
                
                # Second derivatives (acceleration)
                yellow_left_acceleration = round((yellow_left_velocity - yellow_left_velocity_1) / self.dt, 1)
                pale_left_acceleration = round((pale_left_velocity - pale_left_velocity_1) / self.dt, 1)
                yellow_right_acceleration = round((yellow_right_velocity - yellow_right_velocity_1) / self.dt, 1)
                pale_right_acceleration = round((pale_right_velocity - pale_right_velocity_1) / self.dt, 1)

                dark_left_acceleration = round((dark_left_velocity - dark_left_velocity_1) / self.dt)
                dark_right_acceleration = round((dark_right_velocity - dark_right_velocity_1) / self.dt)
            
            if len(self.vision_memory_acc) >= 1:
                yellow_left_acceleration = round(yellow_left_acceleration )
                pale_left_acceleration = round(pale_left_acceleration)
                yellow_right_acceleration = round(yellow_right_acceleration)
                pale_right_acceleration = round(pale_right_acceleration)

                dark_left_acceleration = round(dark_left_acceleration)
                dark_right_acceleration = round(dark_right_acceleration)

                
            # Normal vision response
            mean_gradient = left_eye.mean() - right_eye.mean() # if gradient > 0 -> less obstacle on left -> turn left

            mean_obstacle = fly_vision.mean() # lower == obstacles, few obstacles -> 0.48-0.59, no obstacle -> 0.50

            vision_speed = np.tanh((mean_obstacle-0.4)/(0.5-0.4))*3/4 +1/4 # function to keep high speed for x < soft threshold, but low speed after
            gradient_bias = np.tanh((100*mean_gradient)**3) # function to turn little when medium/low but turn way more when bigger
            turning_bias = np.tanh(gradient_bias - (dark_left-dark_right)/5)

            vision_response = [vision_speed, turning_bias]

            # Vision change response 
            if len(self.vision_memory_pos) >= 1: 
                left_eye_change = (yellow_left_velocity + pale_left_velocity)/2 # obstacle approaching -> velocity < 0
                right_eye_change = (yellow_right_velocity + pale_right_velocity)/2
                change_gradient = left_eye_change - right_eye_change # if right obstacle approaching -> gradient > 0 -> turn left
                turning_bias += np.tanh(change_gradient/50)


            vision_updated = obs.get("vision_updated", False)
            if vision_updated:
                # Append to memory + deletion
                self.vision_memory_pos.append([yellow_left, pale_left, yellow_right, pale_right, dark_left, dark_right]) 
                if len(self.vision_memory_pos) > 3:
                    self.vision_memory_pos.pop(0)

                if len(self.vision_memory_pos) >= 2:
                    self.vision_memory_vel.append([yellow_left_velocity, pale_left_velocity, yellow_right_velocity, pale_right_velocity, dark_left_velocity, dark_right_velocity])
                if len(self.vision_memory_vel) > 2:
                    self.vision_memory_vel.pop(0)

                if len(self.vision_memory_vel) >= 2:
                    self.vision_memory_acc.append([yellow_left_acceleration, pale_left_acceleration, yellow_right_acceleration, pale_right_acceleration, dark_left_acceleration, dark_right_acceleration])
                if len(self.vision_memory_acc) > 3:
                    self.vision_memory_acc.pop(0)

            return vision_response


    def get_threat_response(self, obs: Observation):
        threat_threshold = -100000 # if mean of last 3 obs < threshold -> ball looming (for pale ommatidia)
        action = [0, 0]

        if len(self.vision_memory_acc) >= 1:
            looming = np.array(self.vision_memory_acc)
            changing = np.array(self.vision_memory_vel)

            # left and right looming with pale ommatidia
            left_looming = np.mean(looming[-2:, 1])
            right_looming = np.mean(looming[-2:, 3])

            continuous_left_looming = np.all(changing[-2:, 4] >= 10000)
            continuous_right_looming = np.all(changing[-2:, 5] >= 10000)


            if left_looming < threat_threshold and continuous_left_looming:
                action = [-1, 0]
            
            if right_looming < threat_threshold and continuous_right_looming:
                action = [-1, 0]

        return action
    
    def get_home_controller(self, obs: Observation):        
        # Homing vector calculation
        to_home = self.home_position - self.position
        # Calculation of the heading angle to the home position
        target_heading = np.arctan2(to_home[1], to_home[0])
        # Calculate the heading error (difference between the target heading and the current heading)
        err = (target_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
        # Calculate the distance to the home position
        dist = np.linalg.norm(to_home)

        # Implementation of the return controller
        if np.abs(err) > np.deg2rad(45): # hard coding of a left or right turn to face the home position
            if err > 0:
                action = np.array([-1.0, 1.0])  # turn left
            else:
                action = np.array([1.0, -1.0])  # turn right
        else: # proportional controller
            turning_bias = np.tanh(err/(np.pi/4)) # tanh function to limit the turning bias
            action = np.array(np.clip([1-turning_bias, 1+turning_bias], 0, 1))

        # If the distance to the home position is less than 1mm, stop the simulation
        if dist <= 1:
            self.quit = True
            print("Nest return completed.")

        return action
            
        # joints, adhesion = step_cpg(
        #     cpg_network=self.cpg_network,
        #     preprogrammed_steps=self.preprogrammed_steps,
        #     action=action,
        # )

        # return {"joints": joints, "adhesion": adhesion}
    
    def get_blocking_response(self, obs: Observation, odor_bias):
        delta = np.linalg.norm(self.history[-1] - self.position)
        if delta < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= self.stuck_threshold:
            print("Stuck : step back")
            self.stuck_counter = 0
            self.recovery_steps = 2000

        if getattr(self, "recovery_steps", 0) > 0:
            action = np.clip([-1.0-odor_bias/2, -1.0+odor_bias/2], -1, 0)
            self.recovery_steps -= 1
            return action
        else:
            return [0, 0]

            # joints, adhesion = step_cpg(
            #     cpg_network=self.cpg_network,
            #     preprogrammed_steps=self.preprogrammed_steps,
            #     action=action,
            # )
            # return {"joints": joints, "adhesion": adhesion}


    def get_actions(self, obs: Observation):
        # Update the position and heading filters
        v_filt = self.filter_velocity(obs)
        h_filt = self.filter_heading(obs)
        self.update_position_filters(v_filt, h_filt)

        # Reached odor, returning home
        if obs.get("reached_odour", False):
            self.returning_home = True

        if self.returning_home:
            action = self.get_home_controller(obs)
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=action,
            )
            return {
                "joints": joint_angles,
                "adhesion": adhesion,
            }

        # --- Logique CPG (odor, vision, threat) ---
        odor_action   = self.get_odor_bias(obs)
        vision_action = self.get_vision_bias(obs)
        threat_action = self.get_threat_response(obs)   

        block_action = self.get_blocking_response(obs, odor_action[1])   
        
        # Odor + vision response
        odor_power = np.clip(1-odor_action[0], 0, 1) # power of speed
        vision_power = np.clip(np.abs(vision_action[1]), 0, 1) # power of turning

        weight_speed = (vision_power - odor_power/2 + 1/2)*2/3 # if vision>>odor -> 1, if odor>>vision -> 0
        weight_turning = vision_power

        speed = weight_speed*vision_action[0] + (1-weight_speed)*odor_action[0]
        turning = weight_turning*vision_action[1] + (1-weight_turning)*odor_action[1]

        # Threat response
        if threat_action != [0, 0]:
            speed, turning = threat_action

        # Anti-blocking response
        if block_action != [0, 0]:
            speed, turning = block_action
          
        # Final action
        action = np.array([speed-turning/2, speed+turning/2])
        action = np.clip(action, -1, 1)

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation) -> bool:
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.position = np.zeros(2)
        self.heading = 0.0
        self.history = [self.position.copy()]
        self.real_history = []  
        self.returning_home = False
        self.angle_error_prev = 0.0
        self.dist_error_prev  = 0.0
