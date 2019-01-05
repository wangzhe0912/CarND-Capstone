#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
from light_classification.tl_detector import TrafficLightDetector
from light_classification.tl_detector_single_shot import TrafficLightDetectorSingleShot

import tf
import cv2
import yaml
import math

import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=52428800)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.on_site = rospy.get_param('/tl_detector/on_site', False)
        rospy.logwarn("Setup is on site: {}".format(self.on_site))

        if not self.on_site:
            self.light_classifier = TLClassifier()
            self.light_detector = TrafficLightDetector(0.7)
            self.light_detector.get_detection(np.zeros((600, 800, 3)).astype(np.uint8))

        else:
            # TODO single shot detection and classification
            self.light_detector_single_shot = TrafficLightDetectorSingleShot(0.3)
            self.light_detector_single_shot.get_tl_type(np.zeros((1096, 1368, 3)).astype(np.uint8))

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.class_map = ["red", "yellow", "green", "unused", "unk"]

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def compute_dist(self, pose_a, pose_b):
        """Computes the distance between two given positions
        :param pose_a:
        :param pose_b:
        :return: Euclidean distance between two positions
        """

        x_a = pose_a.position.x
        y_a = pose_a.position.y
        x_b = pose_b.position.x
        y_b = pose_b.position.y

        return math.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        if self.waypoints is None:
            return -1

        min_dist = 1e4
        closest_wp = 1e4

        for i, wp in enumerate(self.waypoints):
            dist = self.compute_dist(pose, wp.pose.pose)

            if dist < min_dist:
                min_dist = dist
                closest_wp = i

        return closest_wp

    def _get_light_state_simulator(self, light):
        if self.light_detector is None or self.light_classifier is None:
            return TrafficLight.RED

        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8"), cv2.COLOR_BGR2RGB)

        # get tl detections
        detections, scores = self.light_detector.get_detection(cv_image)

        # Get classification
        tl_type = self.light_classifier.get_classification(cv_image, detections)

        return tl_type

    def _get_light_state_on_site(self, light):
        if self.light_detector_single_shot is None:
            return TrafficLight.RED

        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8"), cv2.COLOR_BGR2RGB)

        # get tl detections

        tl_type, det, score = self.light_detector_single_shot.get_tl_type(cv_image)
        rospy.logwarn("Traffic light {} {} {} {}".format(tl_type, self.class_map[int(tl_type)],  det, score))

        return tl_type


    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        if not self.on_site:
            return self._get_light_state_simulator(light)

        return self._get_light_state_on_site(light)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None
        light_wp = None
        dist_to_light = 1e4

        # state = self.get_light_state(light)

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)
            # rospy.logerr('Car pos: {}'.format(car_position))
        else:
            return -1, TrafficLight.UNKNOWN

        #TODO find the closest visible traffic light (if one exists)
        for stop_line_position in stop_line_positions:
            stop_line_pose = Pose()
            stop_line_pose.position.x = stop_line_position[0]
            stop_line_pose.position.y = stop_line_position[1]

            # get the closest wp
            stop_line_wp = self.get_closest_waypoint(stop_line_pose)

            # check is stop line is in front of the car
            if stop_line_wp >= car_position:
                if not light_wp or stop_line_wp < light_wp:
                    light_wp = stop_line_wp
                    light = stop_line_pose

        if car_position and light_wp:
            dist_to_light = abs(car_position - light_wp)

        # check the tl state if it's within 100 wps
        if light and dist_to_light < 100:
            state = self.get_light_state(light)
            return light_wp, state

        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')