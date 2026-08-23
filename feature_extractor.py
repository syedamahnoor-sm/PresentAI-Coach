import math


# =========================================================
# BASIC HELPERS
# =========================================================

def safe_divide(a, b):
    """
    Safely divide two values and avoid division-by-zero.
    """
    denominator = max(abs(b), 1e-6)
    return a / denominator


def distance_2d(point_a, point_b):
    """
    Euclidean distance using x and y.
    """
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
    )


def distance_3d(point_a, point_b):
    """
    Euclidean distance using x, y and z.
    """
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
        + (point_a[2] - point_b[2]) ** 2
    )


def midpoint(point_a, point_b):
    """
    Midpoint between two 3D points.
    """
    return [
        (point_a[0] + point_b[0]) / 2,
        (point_a[1] + point_b[1]) / 2,
        (point_a[2] + point_b[2]) / 2,
    ]


def landmark_to_list(landmark):
    """
    Convert a MediaPipe face landmark object into [x, y, z].
    """
    return [
        landmark.x,
        landmark.y,
        landmark.z
    ]


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(
    pose_landmarks,
    world_landmarks,
    face_landmarks=None
):
    """
    Extract posture-related numerical features.

    Pose landmarks are REQUIRED.

    Face landmarks are OPTIONAL so side/profile views
    are still usable when Face Landmarker cannot detect
    the full face.

    Parameters
    ----------
    pose_landmarks:
        [[x, y, z], ...] normalized pose landmarks.

    world_landmarks:
        [[x, y, z], ...] MediaPipe 3D world landmarks.

    face_landmarks:
        MediaPipe Face Landmarker output, or None.

    Returns
    -------
    Dictionary of ML-ready numerical features.
    """

    if pose_landmarks is None or world_landmarks is None:
        return None

    if len(pose_landmarks) < 25 or len(world_landmarks) < 25:
        return None


    # =====================================================
    # POSE LANDMARKS
    # =====================================================

    nose_pose = pose_landmarks[0]

    left_ear_pose = pose_landmarks[7]
    right_ear_pose = pose_landmarks[8]

    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]


    # =====================================================
    # WORLD LANDMARKS
    # =====================================================

    nose_world = world_landmarks[0]

    left_shoulder_world = world_landmarks[11]
    right_shoulder_world = world_landmarks[12]

    left_hip_world = world_landmarks[23]
    right_hip_world = world_landmarks[24]


    # =====================================================
    # REFERENCE MEASUREMENTS
    # =====================================================

    shoulder_width_2d = distance_2d(
        left_shoulder,
        right_shoulder
    )

    shoulder_width_3d = distance_3d(
        left_shoulder_world,
        right_shoulder_world
    )

    shoulder_width_2d = max(
        shoulder_width_2d,
        1e-6
    )

    shoulder_width_3d = max(
        shoulder_width_3d,
        1e-6
    )


    shoulder_center = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_center = midpoint(
        left_hip,
        right_hip
    )


    shoulder_center_world = midpoint(
        left_shoulder_world,
        right_shoulder_world
    )

    hip_center_world = midpoint(
        left_hip_world,
        right_hip_world
    )


    # =====================================================
    # 1. SHOULDER TILT
    # =====================================================

    shoulder_dx = abs(
        right_shoulder[0]
        - left_shoulder[0]
    )

    shoulder_dy = abs(
        right_shoulder[1]
        - left_shoulder[1]
    )

    shoulder_tilt = math.degrees(
        math.atan2(
            shoulder_dy,
            max(shoulder_dx, 1e-6)
        )
    )


    # =====================================================
    # 2. SHOULDER DEPTH ASYMMETRY
    # =====================================================

    shoulder_depth_difference = safe_divide(
        left_shoulder_world[2]
        - right_shoulder_world[2],
        shoulder_width_3d
    )


    # =====================================================
    # 3. SIDEWAYS TORSO LEAN
    # =====================================================

    torso_dx = (
        shoulder_center_world[0]
        - hip_center_world[0]
    )

    torso_dy = (
        shoulder_center_world[1]
        - hip_center_world[1]
    )

    torso_side_angle = math.degrees(
        math.atan2(
            abs(torso_dx),
            max(abs(torso_dy), 1e-6)
        )
    )


    # Normalized left/right displacement
    torso_side_offset = safe_divide(
        shoulder_center[0]
        - hip_center[0],
        shoulder_width_2d
    )


    # =====================================================
    # 4. FORWARD / BACKWARD TORSO ORIENTATION
    # =====================================================

    torso_dz = (
        shoulder_center_world[2]
        - hip_center_world[2]
    )

    # Signed angle:
    # useful because forward/backward are different states.
    torso_depth_angle = math.degrees(
        math.atan2(
            torso_dz,
            max(abs(torso_dy), 1e-6)
        )
    )

    torso_depth_angle_abs = abs(
        torso_depth_angle
    )


    # =====================================================
    # 5. TORSO LENGTH / COMPRESSION
    # =====================================================

    torso_length = distance_3d(
        shoulder_center_world,
        hip_center_world
    )

    torso_length_ratio = safe_divide(
        torso_length,
        shoulder_width_3d
    )


    # 2D torso height also provides a front-camera cue
    torso_vertical_distance = abs(
        shoulder_center[1]
        - hip_center[1]
    )

    torso_vertical_ratio = safe_divide(
        torso_vertical_distance,
        shoulder_width_2d
    )


    # =====================================================
    # 6. HEAD POSITION RELATIVE TO TORSO
    # =====================================================

    head_horizontal_offset = safe_divide(
        nose_pose[0]
        - shoulder_center[0],
        shoulder_width_2d
    )

    head_vertical_distance = safe_divide(
        shoulder_center[1]
        - nose_pose[1],
        shoulder_width_2d
    )

    head_depth_offset = safe_divide(
        nose_world[2]
        - shoulder_center_world[2],
        shoulder_width_3d
    )


    # =====================================================
    # 7. POSE-BASED HEAD TILT
    # =====================================================

    ear_dx = abs(
        right_ear_pose[0]
        - left_ear_pose[0]
    )

    ear_dy = abs(
        right_ear_pose[1]
        - left_ear_pose[1]
    )

    pose_head_tilt = math.degrees(
        math.atan2(
            ear_dy,
            max(ear_dx, 1e-6)
        )
    )


    # =====================================================
    # POSE-ONLY FEATURES
    # =====================================================

    features = {
        "shoulder_tilt":
            shoulder_tilt,

        "shoulder_depth_difference":
            shoulder_depth_difference,

        "torso_side_angle":
            torso_side_angle,

        "torso_side_offset":
            torso_side_offset,

        "torso_depth_angle":
            torso_depth_angle,

        "torso_depth_angle_abs":
            torso_depth_angle_abs,

        "torso_length_ratio":
            torso_length_ratio,

        "torso_vertical_ratio":
            torso_vertical_ratio,

        "head_horizontal_offset":
            head_horizontal_offset,

        "head_vertical_distance":
            head_vertical_distance,

        "head_depth_offset":
            head_depth_offset,

        "pose_head_tilt":
            pose_head_tilt,
    }


    # =====================================================
    # FACE OPTIONAL
    # =====================================================

    face_available = (
        face_landmarks is not None
        and len(face_landmarks) > 454
    )

    features["face_available"] = (
        1 if face_available else 0
    )


    # =====================================================
    # DEFAULT FACE VALUES
    # =====================================================
    #
    # NaN means:
    # "This measurement was unavailable."
    #
    # Later, our sklearn pipeline will impute missing
    # values properly instead of treating them as zero.
    # =====================================================

    features["face_head_tilt"] = math.nan
    features["face_height_ratio"] = math.nan
    features["chin_to_shoulder_distance"] = math.nan
    features["cheek_depth_difference"] = math.nan
    features["eye_depth_difference"] = math.nan
    features["face_horizontal_offset"] = math.nan


    if not face_available:
        return features


    # =====================================================
    # FACE LANDMARKS
    # =====================================================

    nose_face = landmark_to_list(
        face_landmarks[1]
    )

    forehead = landmark_to_list(
        face_landmarks[10]
    )

    chin = landmark_to_list(
        face_landmarks[152]
    )

    left_eye = landmark_to_list(
        face_landmarks[33]
    )

    right_eye = landmark_to_list(
        face_landmarks[263]
    )

    left_cheek = landmark_to_list(
        face_landmarks[234]
    )

    right_cheek = landmark_to_list(
        face_landmarks[454]
    )


    # =====================================================
    # 8. FACE WIDTH
    # =====================================================

    face_width = distance_2d(
        left_cheek,
        right_cheek
    )

    face_width = max(
        face_width,
        1e-6
    )


    # =====================================================
    # 9. FACE / HEAD TILT
    # =====================================================

    eye_dx = abs(
        right_eye[0]
        - left_eye[0]
    )

    eye_dy = abs(
        right_eye[1]
        - left_eye[1]
    )

    face_head_tilt = math.degrees(
        math.atan2(
            eye_dy,
            max(eye_dx, 1e-6)
        )
    )


    # =====================================================
    # 10. FACE HEIGHT RATIO
    # =====================================================

    face_height = distance_2d(
        forehead,
        chin
    )

    face_height_ratio = safe_divide(
        face_height,
        face_width
    )


    # =====================================================
    # 11. CHIN / SHOULDER RELATION
    # =====================================================

    chin_to_shoulder_distance = safe_divide(
        shoulder_center[1]
        - chin[1],
        shoulder_width_2d
    )


    # =====================================================
    # 12. FACE ORIENTATION / DEPTH
    # =====================================================

    cheek_depth_difference = (
        left_cheek[2]
        - right_cheek[2]
    )

    eye_depth_difference = (
        left_eye[2]
        - right_eye[2]
    )


    # =====================================================
    # 13. FACE CENTER OFFSET
    # =====================================================

    face_horizontal_offset = safe_divide(
        nose_face[0]
        - shoulder_center[0],
        shoulder_width_2d
    )


    # =====================================================
    # ADD FACE FEATURES
    # =====================================================

    features.update({
        "face_head_tilt":
            face_head_tilt,

        "face_height_ratio":
            face_height_ratio,

        "chin_to_shoulder_distance":
            chin_to_shoulder_distance,

        "cheek_depth_difference":
            cheek_depth_difference,

        "eye_depth_difference":
            eye_depth_difference,

        "face_horizontal_offset":
            face_horizontal_offset,
    })


    return features