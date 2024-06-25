import subprocess

audio_path = "videobot\welcome.mp3"
video_path = "videobot\output_video.avi"  # Create a blank .avi beforehand
output_path = "videobot\Videobot_generated_video.avi"

# FFmpeg command to combine audio and video
command = ["ffmpeg", "-i", audio_path, "-i", video_path, "-c:v", "copy", "-c:a", "copy", output_path]

subprocess.run(command)