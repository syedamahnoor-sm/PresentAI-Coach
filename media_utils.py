import os
import subprocess
import tempfile


def extract_audio_from_video(
    video_path,
    sample_rate=16000
):
    """
    Extract mono WAV audio from a video using FFmpeg.

    Returns:
        temporary WAV file path
    """

    if not os.path.exists(
        video_path
    ):

        raise FileNotFoundError(
            video_path
        )


    temp_file = (
        tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
    )

    audio_path = (
        temp_file.name
    )

    temp_file.close()


    command = [
        "ffmpeg",

        "-y",

        "-i",
        video_path,

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(
            sample_rate
        ),

        "-acodec",
        "pcm_s16le",

        audio_path,
    ]


    try:

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    except FileNotFoundError:

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )

        raise RuntimeError(
            "FFmpeg was not found. "
            "Install FFmpeg or add it to PATH."
        )


    except subprocess.CalledProcessError:

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )

        raise RuntimeError(
            "Could not extract audio "
            "from video."
        )


    return audio_path