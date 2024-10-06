import rosbag
import cv2
import os
from cv_bridge import CvBridge
from rosbag.bag import ROSBagUnindexedException

def extract_images_from_bag(bag_file, output_dir, interval=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        bag = rosbag.Bag(bag_file, 'r')
    except ROSBagUnindexedException:
        print(f"Bag file {bag_file} is unindexed.")
        #os.system(f"rosbag reindex {bag_file}")
        bag = rosbag.Bag(bag_file, 'r')

    bridge = CvBridge()

    image_topics = ['/camera/image_color', '/swir_camera/image_raw']  # Replace with your actual image topics

    last_saved_time = None

    for topic, msg, t in bag.read_messages(topics=image_topics):
        if last_saved_time is None or (t.to_sec() - last_saved_time >= interval):
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = t.to_nsec()
            image_file = os.path.join(output_dir, f"{topic.replace('/', '_')}_{timestamp}.png")
            cv2.imwrite(image_file, cv_image)
            last_saved_time = t.to_sec()

    bag.close()

bag_file = '/home/michal/Documents/datacollections/cheesboard_collection.bag'
output_dir = '/home/michal/Documents/datacollections/extracted_images'
interval = 0.2  # Set the interval in seconds
extract_images_from_bag(bag_file, output_dir, interval)
