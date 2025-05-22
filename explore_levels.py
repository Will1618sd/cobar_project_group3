import argparse
import numpy as np
import cv2
import tqdm
from cobar_miniproject import levels
from cobar_miniproject.keyboard_controller import KeyBoardController
from cobar_miniproject.cobar_fly import CobarFly
from cobar_miniproject.vision import (
    get_fly_vision,
    get_fly_vision_raw,
    render_image_with_vision,
)
from flygym import YawOnlyCamera, SingleFlySimulation
from flygym.arena import FlatTerrain

# OPTIONS
# what to display as the simulation is running
ONLY_CAMERA = 0
WITH_FLY_VISION = 1
WITH_RAW_VISION = 2

VISUALISATION_MODE = WITH_FLY_VISION
# VISUALISATION_MODE = WITH_RAW_VISION

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "--level", type=int, default=4, help="Level to load (default: -1)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the simulation (default: 0)"
    )
    args = parser.parse_args()

    level = args.level
    seed = args.seed
    timestep = 1e-4

    # you can pass in parameters to enable different senses here
    fly = CobarFly(debug=False, enable_vision=True, render_raw_vision=False)

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)

    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_back_track_game",
        targeted_fly_names=[fly.name],
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    controller = KeyBoardController(timestep=timestep, seed=seed)

    # run cpg simulation
    obs, info = sim.reset()
    obs_hist = []
    info_hist = []

    # create window
    cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)

    with tqdm.tqdm(desc="running simulation") as progress_bar:
        while True:
            # Get observations
            obs, reward, terminated, truncated, info = sim.step(
                controller.get_actions(obs)
            )
            if controller.done_level(obs, seed, level):
                # finish the path integration level
                break

            # vision_updated = obs.get("vision_updated", False)
            # if vision_updated:

            #     fly_vision = obs["vision"]
            #     left_eye = fly_vision[0,:,:]
            #     right_eye = fly_vision[1,:,:]

            #     # Yellow : dark
            #     yellow_left = left_eye[:,0].mean()
            #     yellow_right = right_eye[:,0].mean()
                
            #     # Pale : light
            #     pale_left = left_eye[:,1].mean()
            #     pale_right = right_eye[:,1].mean()

            #     yellow_gradient = yellow_left - yellow_right
            #     pale_gradient = pale_left - pale_right

            #     vision_updated = obs.get("vision_updated", False)
            #     if vision_updated:
            #         print("yellow gradient", np.round(yellow_gradient,4))
            #         print("pale gradient  ", np.round(pale_gradient,4))

            obs_ = obs.copy()
            if not obs_["vision_updated"]:
                if "vision" in obs_:
                    del obs_["vision"]
                if "raw_vision" in obs_:
                    del obs_["raw_vision"]
            obs_hist.append(obs_)
            info_hist.append(info)

            rendered_img = sim.render()[0]
            if rendered_img is not None:
                if VISUALISATION_MODE == WITH_FLY_VISION:
                    rendered_img = render_image_with_vision(
                        rendered_img, get_fly_vision(fly), obs["odor_intensity"],
                    )
                elif VISUALISATION_MODE == WITH_RAW_VISION:
                    rendered_img = render_image_with_vision(
                        rendered_img, get_fly_vision_raw(fly), obs["odor_intensity"],
                    )
                rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
                cv2.imshow("Simulation", rendered_img)
                cv2.waitKey(1)

            if controller.quit:
                print("Simulation terminated by user.")
                break
            if hasattr(level_arena, "quit") and level_arena.quit:
                print("Target reached. Simulation terminated.")
                break

            progress_bar.update()

    print("Simulation finished")

    # Save video
    cam.save_video("./outputs/hybrid_controller.mp4", 0)