
filepath = "/mnt/spaceai-internal/panda-intervid/untar_data/disk2/panda-000-137/nvme/tmp/heyinan/panda/079/JUWYUZbdRXk-0:04:19.158-0:04:30.570.mp4"
filename = filepath.split('/')[-1]  # "JUWYUZbdRXk-0:04:19.158-0:04:30.570.mp4"
print(filename)
video_id = filename[:11]  # "JUWYUZbdRXk"
print(video_id)