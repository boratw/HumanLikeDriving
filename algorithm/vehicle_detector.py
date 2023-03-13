import numpy as np
import math
import carla
from shapely.geometry import Polygon

def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Check if a location is both within a certain distance from a reference object.
    By using 'angle_interval', the angle between the location and reference transform
    will also be tkaen into account, being 0 a location in front and 180, one behind.

    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    # Further than the max distance
    if norm_target > max_distance:
        return False

    # We don't care about the angle, nothing else to check
    if not angle_interval:
        return True

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return min_angle < angle < max_angle



def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def detect_vehicles(actor, map, route, others, max_distance, up_angle_th=90, low_angle_th=0, lane_offset=0):

    

    ego_transform = actor.get_transform()
    ego_wpt = map.get_waypoint(actor.get_location())

    # Get the right offset
    if ego_wpt.lane_id < 0 and lane_offset != 0:
        lane_offset *= -1

    # Get the transform of the front of the ego
    ego_forward_vector = ego_transform.get_forward_vector()
    ego_extent = actor.bounding_box.extent.x
    ego_front_transform = ego_transform
    ego_front_transform.location += carla.Location(
        x=ego_extent * ego_forward_vector.x,
        y=ego_extent * ego_forward_vector.y,
    )

    for target_vehicle in others:
        target_transform = target_vehicle.get_transform()
        target_wpt = map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

        # Simplified version for outside junctions
        if not ego_wpt.is_junction or not target_wpt.is_junction:

            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                continue

            target_forward_vector = target_transform.get_forward_vector()
            target_extent = target_vehicle.bounding_box.extent.x
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                x=target_extent * target_forward_vector.x,
                y=target_extent * target_forward_vector.y,
            )

            if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        # Waypoints aren't reliable, check the proximity of the vehicle to the route
        else:
            route_bb = []
            ego_location = ego_transform.location
            extent_y = actor.bounding_box.extent.y
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            route_bb.append([p1.x, p1.y, p1.z])
            route_bb.append([p2.x, p2.y, p2.z])

            for tr in route:
                if ego_location.distance(tr.location) > max_distance:
                    break

                r_vec = tr.get_right_vector()
                p1 = tr.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = tr.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

            if len(route_bb) < 3:
                # 2 points don't create a polygon, nothing to check
                return (False, None, -1)
            ego_polygon = Polygon(route_bb)

            # Compare the two polygons
            for target_vehicle in others:
                target_extent = target_vehicle.bounding_box.extent.x
                if target_vehicle.id == actor.id:
                    continue
                if ego_location.distance(target_vehicle.get_location()) > max_distance:
                    continue

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if ego_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            return (False, None, -1)

    return (False, None, -1)

