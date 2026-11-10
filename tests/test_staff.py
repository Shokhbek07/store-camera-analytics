"""Staff versus customer: uniform colour plus time spent in the staff zone."""

import cv2
import numpy as np

from storecam.events import ZoneSampler
from storecam.staff import StaffClassifier, colour_distance, torso_lab

UNIFORM_LAB = (113, 129, 141)  # the beige measured on counter_1650
STAFF_ZONE = [(0, 0), (100, 0), (100, 200), (0, 200)]
QUEUE_ZONE = [(100, 0), (300, 0), (300, 200), (100, 200)]


def bgr(lab):
    return cv2.cvtColor(np.uint8([[lab]]), cv2.COLOR_LAB2BGR)[0, 0].tolist()


BEIGE, DARK = bgr(UNIFORM_LAB), bgr((60, 127, 130))


def frame(people):
    """people: list of (x_centre, colour); every person box is 40 wide and 120 tall, feet at y=150."""
    img = np.full((200, 300, 3), 200, np.uint8)
    for x, colour in people:
        cv2.rectangle(img, (x - 20, 30), (x + 20, 150), colour, -1)
    return img


def box(x):
    return (x - 20, 30, x + 20, 150)


def test_torso_colour_of_a_uniform_matches_the_reference():
    c = torso_lab(frame([(50, BEIGE)]), box(50))
    assert colour_distance(c, UNIFORM_LAB) < 3
    assert colour_distance(torso_lab(frame([(50, DARK)]), box(50)), UNIFORM_LAB) > 12


def test_uniform_that_was_behind_the_counter_is_staff_in_the_queue():
    s = StaffClassifier(UNIFORM_LAB, 12, STAFF_ZONE, min_zone_seconds=1.0)
    for i in range(15):  # 1.5 s behind the counter
        s.update(frame([(50, BEIGE)]), {1: box(50)}, t=i / 10)
    assert s.update(frame([(200, BEIGE)]), {1: box(200)}, t=1.6) == {1}


def test_uniform_colour_never_in_the_staff_zone_is_a_customer():
    s = StaffClassifier(UNIFORM_LAB, 12, STAFF_ZONE, min_zone_seconds=1.0)
    for i in range(30):  # a customer in a beige coat, only ever in the queue
        staff = s.update(frame([(200, BEIGE)]), {1: box(200)}, t=i / 10)
    assert staff == set()


def test_customer_reaching_into_the_staff_zone_is_not_staff():
    s = StaffClassifier(UNIFORM_LAB, 12, STAFF_ZONE, min_zone_seconds=1.0)
    for i in range(30):
        staff = s.update(frame([(50, DARK)]), {1: box(50)}, t=i / 10)
    assert staff == set()


def test_forget_drops_history_of_dead_tracks():
    s = StaffClassifier(UNIFORM_LAB, 12, STAFF_ZONE, min_zone_seconds=1.0)
    for i in range(15):
        s.update(frame([(50, BEIGE)]), {1: box(50)}, t=i / 10)
    s.forget(alive=set())
    assert s.update(frame([(200, BEIGE)]), {1: box(200)}, t=2.0) == set()


def test_zone_sampler_leaves_staff_out_of_the_queue_count():
    z = ZoneSampler(QUEUE_ZONE, STAFF_ZONE)
    z.update([(200, 150), (250, 150), (50, 150)], 0.0, skip=[True, False, True])
    (sample,) = z.update([], 1.0)
    assert (sample.queue, sample.staff_present) == (1, True)
