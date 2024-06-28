from moviepy.editor import VideoFileClip, AudioFileClip

def merge_audio_video(video_file, audio_file, output_file):
    # Load the video file
    video_clip = VideoFileClip(video_file)
    
    # Load the audio file
    audio_clip = AudioFileClip(audio_file)
    
    # Set the audio of the video clip
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write the output to a file
    video_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

# Example usage
video_file = 'videobot\Only_video.avi'
audio_file = 'videobot\Audio.mp3'
output_file = 'videobot\merged_video.mp4'

merge_audio_video(video_file, audio_file, output_file)
