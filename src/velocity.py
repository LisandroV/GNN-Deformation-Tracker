import numpy as np
import data.test_data as Data

level_set = np.array(Data.level_set) # shape: (100, 47, 2)
finger = np.array(Data.finger_data) # shape: (100, 4)

velocity_level_set = []


TIME_STEPS = 100
CONTROL_POINTS = 47
DIMENSIONS = 2

first_polygon = np.append(level_set[0,:,:], [[0,0]]*47, axis=1) # first polygon with velocity=0
new_polygons = [first_polygon]

# VELOCITY ON CONTROL POINTS
for time in range(1,TIME_STEPS):
    current_polygon = level_set[time,:,:]
    previous_polygon = new_polygons[time-1][:,:2] # coordinates without velocity
    velocities = current_polygon - previous_polygon
    new_velocity_polygon = np.append(current_polygon, velocities, axis=1) # current polygon with velocity
    new_polygons.append(new_velocity_polygon)

# New Polygon with velocity: [[ x, y, x_velocity, y_velocity], ...]
polygons_with_velocity = np.array(new_polygons) # shape: (100, 47, 4)

print("level_set = " + str(polygons_with_velocity.tolist()))

# VELOCITY ON ROBOTIC FINGER

velocities = [[0, 0]]
for time in range(1,TIME_STEPS):
    previous = finger[time-1,:2]
    current = finger[time,:2]
    velocities.append(current - previous)

np_velocities = np.array(velocities)

# New finger with velocity: [[ x, y, force_x, force_y, velocity_x, velocity_y], ...]
finger_with_velocity = np.append(finger, np_velocities, axis=1)
print("finger_data = " + str(finger_with_velocity.tolist()))

