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

        # Step counter for downsampling plot
        self.step_count = 0
        self.plot_interval = 300      # plot every N steps

        # Initialisation of the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.line_est,)  = self.ax.plot([], [], "-o", label="Estimation")
        (self.line_real,) = self.ax.plot([], [], "-x", label="Real")
        self.real_history = []
        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
            frameon=False,
        )
        self.fig.tight_layout()
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_title('Real vs Estimated Fly Path')
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')


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
    
    def _update_plot(self, obs):
        # Estimated position
        self.history.append(self.position.copy())
        data_est = np.asarray(self.history)
        self.line_est.set_data(data_est[:, 0], data_est[:, 1])

        # Real position
        real = obs.get("debug_fly")
        if real is not None:
            self.real_history.append(real[0, :2].copy())
            data_real = np.asarray(self.real_history)
            self.line_real.set_data(data_real[:, 0], data_real[:, 1])

        # Update plot limits
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

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

                # if (np.abs(odor_gradient) < 1e-5) or (turning_bias/odor_intensity < 100) : # Small gradient or big odor source --> turn less
                #     # Gentle turn
                #     action = [odor_speed, turning_bias]
                # else: 
                #     # Hard turn
                #     action = [0.1, turning_bias*100]

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
       
            # Print if needed
            # vision_updated = obs.get("vision_updated", False)
            # if vision_updated and len(self.odor_memory_vel) >= 2:
            #     print(f"Position : left : {odor_left}, right : {odor_right}, gradient : {odor_gradient}, intensity : {odor_intensity}")
            #     print(f"Velocity : left : {odor_left_velocity}, right : {odor_right_velocity}, gradient : {odor_gradient_velocity}, intensity : {odor_intensity_velocity}")

            return action
            
        else: 
            return [0, 0]  
 

    def get_vision_bias(self, obs: Observation):

        dt = 0.0005 # 5s/10000it
        alpha = 0.7
        
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
                # Filtering
                yellow_left_velocity_1 = self.vision_memory_vel[-1][0]
                pale_left_velocity_1 = self.vision_memory_vel[-1][1]
                yellow_right_velocity_1 = self.vision_memory_vel[-1][2]
                pale_right_velocity_1 = self.vision_memory_vel[-1][3]

                dark_left_velocity_1 = self.vision_memory_vel[-1][4]
                dark_right_velocity_1 = self.vision_memory_vel[-1][5]

                # yellow_left_velocity = round(alpha*yellow_left_velocity + (1-alpha)*yellow_left_velocity_1, 2)
                # pale_left_velocity = round(alpha*pale_left_velocity + (1-alpha)*pale_left_velocity_1, 2)
                # yellow_right_velocity = round(alpha*yellow_right_velocity + (1-alpha)*yellow_right_velocity_1, 2)
                # pale_right_velocity = round(alpha*pale_right_velocity + (1-alpha)*pale_right_velocity_1, 2)

                # dark_left_velocity = round(alpha*dark_left_velocity + (1-alpha)*dark_left_velocity_1)
                # dark_right_velocity = round(alpha*dark_right_velocity + (1-alpha)*dark_right_velocity_1)

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
                # Filtering
                yellow_left_acceleration_1 = self.vision_memory_acc[-1][0]
                pale_left_acceleration_1 = self.vision_memory_acc[-1][1]
                yellow_right_acceleration_1 = self.vision_memory_acc[-1][2]
                pale_right_acceleration_1 = self.vision_memory_acc[-1][3]

                dark_left_acceleration_1 = self.vision_memory_acc[-1][4]
                dark_right_acceleration_1 = self.vision_memory_acc[-1][5]

                # yellow_left_acceleration = round(alpha*yellow_left_acceleration + (1-alpha)*yellow_left_acceleration_1)
                # pale_left_acceleration = round(alpha*pale_left_acceleration + (1-alpha)*pale_left_acceleration_1)
                # yellow_right_acceleration = round(alpha*yellow_right_acceleration + (1-alpha)*yellow_right_acceleration_1)
                # pale_right_acceleration = round(alpha*pale_right_acceleration + (1-alpha)*pale_right_acceleration_1)

                # dark_left_acceleration = round(alpha*dark_left_acceleration + (1-alpha)*dark_left_acceleration_1)
                # dark_right_acceleration = round(alpha*dark_right_acceleration + (1-alpha)*dark_right_acceleration_1)

                yellow_left_acceleration = round(yellow_left_acceleration )
                pale_left_acceleration = round(pale_left_acceleration)
                yellow_right_acceleration = round(yellow_right_acceleration)
                pale_right_acceleration = round(pale_right_acceleration)

                dark_left_acceleration = round(dark_left_acceleration)
                dark_right_acceleration = round(dark_right_acceleration)

                
            # Normal vision response
            yellow_gradient = yellow_left - yellow_right # if gradient > 0 -> less obstacle on left -> turn left
            pale_gradient = pale_left - pale_right
            mean_gradient = left_eye.mean() - right_eye.mean()

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

            # Print if needed
            # vision_updated = obs.get("vision_updated", False)
            # if vision_updated and len(self.vision_memory_vel) >= 2:
                # print(f"Count nb of dark pixels : left : {dark_left}, right : {dark_right}")
                # print(f"\nMean vision : both : {mean_obstacle}, left : {left_eye.mean()}, right : {right_eye.mean()}")
                # print(f"Position gradient : {mean_gradient}")
                # print(f"Position : left - yellow : {yellow_left}, left - pale : {pale_left}, right - yellow : {yellow_right}, right - pale : {pale_right}")
                # print(f"Velocity : left - yellow : {yellow_left_velocity}, left - pale : {pale_left_velocity}, right - yellow : {yellow_right_velocity}, right - pale : {pale_right_velocity}")
                # print(f"Gradient bias : {gradient_bias}, Velocity gradient : {change_gradient}")
                # print(f"Acceleration : left - yellow : {yellow_left_acceleration}, left - pale : {pale_left_acceleration}, right - yellow : {yellow_right_acceleration}, right - pale : {pale_right_acceleration}")
                # print(f"Dark left  : pos : {dark_left}, vel : {dark_left_velocity}, acc : {dark_left_acceleration}")
                # print(f"Dark right : pos : {dark_right}, vel : {dark_right_velocity}, acc : {dark_right_acceleration}")

            return vision_response


    def get_threat_response(self, obs: Observation):
        threat_threshold = -100000 # if mean of last 3 obs < threshold -> ball looming (for pale ommatidia)
        threat_left = False
        threat_right = False
        action = [0, 0]

        if len(self.vision_memory_acc) >= 1:
            yellow_left_looming = self.vision_memory_acc[-1][0]
            pale_left_looming = self.vision_memory_acc[-1][1]
            yellow_right_looming = self.vision_memory_acc[-1][2]
            pale_right_looming = self.vision_memory_acc[-1][3]

            looming = np.array(self.vision_memory_acc)
            changing = np.array(self.vision_memory_vel)

            # left and right looming with pale ommatidia
            left_looming = np.mean(looming[-2:, 1])
            right_looming = np.mean(looming[-2:, 3])

            # continuous_left_looming = np.all(looming[:, 1] < threat_threshold / 100)
            # continuous_right_looming = np.all(looming[:, 3] < threat_threshold / 100)

            continuous_left_looming = np.all(changing[-2:, 4] >= 10000)
            continuous_right_looming = np.all(changing[-2:, 5] >= 10000)


            if left_looming < threat_threshold and continuous_left_looming:
                threat_left = True
                action = [-1, 0]
            
            if right_looming < threat_threshold and continuous_right_looming:
                threat_right = True
                action = [-1, 0]
                

            vision_updated = obs.get("vision_updated", False)
            if vision_updated:
                # print(f"Looming : yellow - left : {yellow_left_looming}, right : {yellow_right_looming}")
                # print(f"Looming : pale - left : {pale_left_looming}, right : {pale_right_looming}")
                # print(f"Right looming : {looming[:, 3]}")
                # print(f"Continuous looming : left : {continuous_left_looming}, right : {continuous_right_looming}")
                if threat_left:
                    print(f"Threat on the left")
                    
                if threat_right:
                    print(f"Threat on the right")
        
        return action

    def get_actions(self, obs: Observation):
        vision_updated = obs.get("vision_updated", False)

        # Update the position and heading filters
        v_filt = self.filter_velocity(obs)
        h_filt = self.filter_heading(obs)
        self.update_position_filters(v_filt, h_filt)

        # Check homing condition
        if obs.get("reached_odour", False):
            self.returning_home = True

        if self.returning_home:
            # Desactivation of the vision (to save memory)
            # Homing vector calculation
            to_home = self.home_position - self.position
            # Calculation of the heading angle to the home position
            target_heading = np.arctan2(to_home[1], to_home[0])
            # Calculate the heading error (difference between the target heading and the current heading)
            err = (target_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
            # Calculate the distance to the home position
            dist = np.linalg.norm(to_home)

            if vision_updated:
                print(f"[Homing] Distance to home: {dist:.2f}, Heading error: {np.rad2deg(err):.2f}°")

            # Implementation of the bang bang control
            # If the heading error is greater than 5 degrees, hard coding of a left or right turn to face the home position
            # else advance straight ahead
            if np.abs(err) > np.deg2rad(45):
                if err > 0:
                    action = np.array([-1.0, 1.0])  # turn left
                else:
                    action = np.array([1.0, -1.0])  # turn right
            else:
                turning_bias = np.tanh(err/(np.pi/4)) # tanh function to limit the turning bias
                action = np.array(np.clip([1-turning_bias, 1+turning_bias], 0, 1))

            # If the distance to the home position is less than 0.5 mm, stop the simulation
            if dist <= 1:
                self.quit = True
                self._update_plot(obs)
                print("Nest return completed.")
                
            joints, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=action,
            )

            self.step_count += 1
            if self.step_count % self.plot_interval == 0:
                self._update_plot(obs)
            return {"joints": joints, "adhesion": adhesion}

        # --- Logique CPG (odor, vision, threat) ---
        odor_action   = self.get_odor_bias(obs)
        vision_action = self.get_vision_bias(obs)
        threat_action = self.get_threat_response(obs)

        odor_grad  = odor_action[0] - odor_action[1] # > 0 if must go right

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
            action = np.array(np.clip([-1.0+odor_grad/2, -1.0-odor_grad/2], -1, 0))
            self.recovery_steps -= 1

            joints, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=action,
            )
            self.step_count += 1
            if self.step_count % self.plot_interval == 0:
                self._update_plot(obs)
            return {"joints": joints, "adhesion": adhesion}
            
        
        odor_power = np.clip(1-odor_action[0], 0, 1) # power of speed
        vision_power = np.clip(np.abs(vision_action[1]), 0, 1) # power of turning

        weight_speed = (vision_power - odor_power/2 + 1/2)*2/3 # if vision>>odor -> 1, if odor>>vision -> 0
        weight_turning = vision_power

        speed = weight_speed*vision_action[0] + (1-weight_speed)*odor_action[0]
        turning = weight_turning*vision_action[1] + (1-weight_turning)*odor_action[1]

        if threat_action != [0, 0]:
            speed, turning = threat_action

        action = np.array([speed-turning/2, speed+turning/2])
        action = np.clip(action, -1, 1)

        # vision_updated = obs.get("vision_updated", False)
        # if vision_updated:
        #     print(f"Odor   : speed : {odor_action[0]}, turning : {odor_action[1]}")
        #     print(f"Vision : speed : {vision_action[0]}, turning : {vision_action[1]}")
        #     # print(f"Action : speed : {speed}, turning : {turning}")
        #     # print(f"Action : left  : {action[0]}, right : {action[1]}")

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action,
        )

        self.step_count += 1
        if self.step_count % self.plot_interval == 0:
            self._update_plot(obs)
        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation, seed=0, level=4) -> bool:
        if self.quit:
            self._update_plot(obs)
            import os
            save_dir = os.getcwd()
            save_path = os.path.join(save_dir, f"outputs/trajectory_seed{seed}_level{level}.png")
            self.fig.savefig(save_path, dpi=300)
            print(f"Graphic saved path: '{save_path}'.")
            # Print the real final position of the fly
            x, y, z = obs["debug_fly"][0]
            print(f"[Real position of the fly:] x = {x:7.2f}  y = {y:7.2f}  z = {z:6.2f} mm", flush=True)
            
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.position = np.zeros(2)
        self.heading = 0.0
        self.step_count = 0
        self.history = [self.position.copy()]
        self.real_history = []  
        self.line_est.set_data([], [])
        self.line_real.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.returning_home = False
        self.angle_error_prev = 0.0
        self.dist_error_prev  = 0.0
