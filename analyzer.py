import math
import statistics


def clamp(value, minimum=0.0, maximum=100.0):
    """Keep a value inside a fixed range."""
    return max(minimum, min(maximum, value))


def midpoint(point_a, point_b):
    """Return the midpoint between two [x, y, z] points."""
    return [
        (point_a[0] + point_b[0]) / 2,
        (point_a[1] + point_b[1]) / 2,
        (point_a[2] + point_b[2]) / 2,
    ]


def continuous_score(deviation, good_limit, bad_limit):
    """
    Convert deviation from calibrated posture into a continuous 0-100 score.

    Small deviation = high score.
    Large deviation = low score.
    """

    if deviation <= good_limit:
        return 100.0

    if deviation >= bad_limit:
        return 0.0

    normalized = (
        (deviation - good_limit)
        / (bad_limit - good_limit)
    )

    return clamp(100 * (1 - normalized))


def extract_posture_metrics(image_landmarks, world_landmarks):
    """
    Extract posture measurements from MediaPipe landmarks.

    image_landmarks:
        Normalized image coordinates used mainly for visual alignment.

    world_landmarks:
        3D coordinates in metres used for real torso geometry.
    """

    # ---------------------------------
    # IMAGE LANDMARKS
    # ---------------------------------

    left_shoulder_img = image_landmarks[11]
    right_shoulder_img = image_landmarks[12]

    left_ear_img = image_landmarks[7]
    right_ear_img = image_landmarks[8]


    # ---------------------------------
    # WORLD LANDMARKS
    # ---------------------------------

    nose_world = world_landmarks[0]

    left_shoulder_world = world_landmarks[11]
    right_shoulder_world = world_landmarks[12]

    left_hip_world = world_landmarks[23]
    right_hip_world = world_landmarks[24]


    shoulder_center_world = midpoint(
        left_shoulder_world,
        right_shoulder_world
    )

    hip_center_world = midpoint(
        left_hip_world,
        right_hip_world
    )


    # =================================
    # 1. SHOULDER TILT
    # =================================

    shoulder_dx = abs(
        right_shoulder_img[0]
        - left_shoulder_img[0]
    )

    shoulder_dy = abs(
        right_shoulder_img[1]
        - left_shoulder_img[1]
    )

    shoulder_tilt_angle = math.degrees(
        math.atan2(
            shoulder_dy,
            max(shoulder_dx, 1e-6)
        )
    )


    # =================================
    # 2. SIDEWAYS TORSO LEAN
    # =================================

    torso_dx = (
        shoulder_center_world[0]
        - hip_center_world[0]
    )

    torso_dy = (
        shoulder_center_world[1]
        - hip_center_world[1]
    )

    sideways_lean_angle = math.degrees(
        math.atan2(
            abs(torso_dx),
            max(abs(torso_dy), 1e-6)
        )
    )


    # =================================
    # 3. FORWARD/BACKWARD TORSO LEAN
    # =================================

    torso_dz = (
        shoulder_center_world[2]
        - hip_center_world[2]
    )

    forward_lean_angle = math.degrees(
        math.atan2(
            abs(torso_dz),
            max(abs(torso_dy), 1e-6)
        )
    )


    # =================================
    # 4. HEAD TILT
    # =================================

    ear_dx = abs(
        right_ear_img[0]
        - left_ear_img[0]
    )

    ear_dy = abs(
        right_ear_img[1]
        - left_ear_img[1]
    )

    head_tilt_angle = math.degrees(
        math.atan2(
            ear_dy,
            max(ear_dx, 1e-6)
        )
    )


    # =================================
    # 5. HEAD-FORWARD POSITION
    # =================================

    shoulder_width_world = math.sqrt(
        (
            left_shoulder_world[0]
            - right_shoulder_world[0]
        ) ** 2
        +
        (
            left_shoulder_world[1]
            - right_shoulder_world[1]
        ) ** 2
        +
        (
            left_shoulder_world[2]
            - right_shoulder_world[2]
        ) ** 2
    )

    shoulder_width_world = max(
        shoulder_width_world,
        0.01
    )

    head_depth_difference = abs(
        nose_world[2]
        - shoulder_center_world[2]
    )

    head_forward_ratio = (
        head_depth_difference
        / shoulder_width_world
    )

    # =================================
    # 6. TORSO UPRIGHTNESS / COMPRESSION
    # =================================
    
    torso_height = math.sqrt(
        (shoulder_center_world[0] - hip_center_world[0]) ** 2
        + (shoulder_center_world[1] - hip_center_world[1]) ** 2
        + (shoulder_center_world[2] - hip_center_world[2]) ** 2
    )
    
    torso_upright_ratio = (
        torso_height / shoulder_width_world
    )
    
    
    return {
        "shoulder_tilt": shoulder_tilt_angle,
        "sideways_lean": sideways_lean_angle,
        "forward_lean": forward_lean_angle,
        "head_tilt": head_tilt_angle,
        "head_forward": head_forward_ratio,
        "torso_upright": torso_upright_ratio,
    }
    
    
def create_posture_baseline(samples):
    """
    Create a user's normal upright posture baseline.

    Median is used instead of mean because it is less affected
    by occasional bad landmark frames.
    """

    return {
        "shoulder_tilt": statistics.median(
            sample["shoulder_tilt"]
            for sample in samples
        ),

        "sideways_lean": statistics.median(
            sample["sideways_lean"]
            for sample in samples
        ),

        "forward_lean": statistics.median(
            sample["forward_lean"]
            for sample in samples
        ),

        "head_tilt": statistics.median(
            sample["head_tilt"]
            for sample in samples
        ),

        "head_forward": statistics.median(
            sample["head_forward"]
            for sample in samples
        ),
        "torso_upright": statistics.median(
            sample["torso_upright"]
            for sample in samples
        ),
    }


def analyze_posture(
    image_landmarks,
    world_landmarks,
    baseline
):
    """
    Compare current posture with calibrated upright posture.
    """

    metrics = extract_posture_metrics(
        image_landmarks,
        world_landmarks
    )


    # ---------------------------------
    # Deviations from user's baseline
    # ---------------------------------

    shoulder_deviation = abs(
        metrics["shoulder_tilt"]
        - baseline["shoulder_tilt"]
    )

    sideways_deviation = abs(
        metrics["sideways_lean"]
        - baseline["sideways_lean"]
    )

    forward_deviation = abs(
        metrics["forward_lean"]
        - baseline["forward_lean"]
    )

    head_tilt_deviation = abs(
        metrics["head_tilt"]
        - baseline["head_tilt"]
    )

    head_forward_deviation = abs(
        metrics["head_forward"]
        - baseline["head_forward"]
    )
    
    torso_upright_deviation = abs(
        metrics["torso_upright"]
        - baseline["torso_upright"]
    )

    # ---------------------------------
    # Continuous scores
    # ---------------------------------

    shoulder_score = continuous_score(
        shoulder_deviation,
        good_limit=2.0,
        bad_limit=12.0
    )

    sideways_score = continuous_score(
        sideways_deviation,
        good_limit=2.0,
        bad_limit=15.0
    )

    forward_score = continuous_score(
        forward_deviation,
        good_limit=3.0,
        bad_limit=22.0
    )

    head_tilt_score = continuous_score(
        head_tilt_deviation,
        good_limit=2.0,
        bad_limit=12.0
    )

    head_forward_score = continuous_score(
        head_forward_deviation,
        good_limit=0.05,
        bad_limit=0.45
    )

    torso_upright_score = continuous_score(
        torso_upright_deviation,
        good_limit=0.05,
        bad_limit=0.35
    )
    
    # ---------------------------------
    # Weighted posture score
    # ---------------------------------

    posture_score = (
        shoulder_score * 0.20
        + sideways_score * 0.20
        + forward_score * 0.30
        + head_tilt_score * 0.15
        + head_forward_score * 0.15
        + torso_upright_score * 0.25
    )


    posture_score = round(
        clamp(posture_score),
        1
    )


    if posture_score >= 90:
        status = "Excellent"

    elif posture_score >= 75:
        status = "Good"

    elif posture_score >= 55:
        status = "Needs Improvement"

    else:
        status = "Poor"


    return {
        "score": posture_score,
        "status": status,

        "shoulder_score": round(shoulder_score, 1),
        "sideways_score": round(sideways_score, 1),
        "forward_score": round(forward_score, 1),
        "head_tilt_score": round(head_tilt_score, 1),
        "head_forward_score": round(head_forward_score, 1),

        "shoulder_tilt": round(
            metrics["shoulder_tilt"],
            2
        ),

        "sideways_lean": round(
            metrics["sideways_lean"],
            2
        ),

        "forward_lean": round(
            metrics["forward_lean"],
            2
        ),

        "head_tilt": round(
            metrics["head_tilt"],
            2
        ),

        "head_forward": round(
            metrics["head_forward"],
            3
        ),
        "torso_upright_score": round(torso_upright_score, 1),
    }