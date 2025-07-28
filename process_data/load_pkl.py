import pickle
import numpy as np
import cv2
from bisect import bisect_left

# ========================= 配置 =========================
PKL_PATH = "factr_teleop/raw_data/trajectory_20250728_105347_0001.pkl"
TIMESTAMP_TOL = 0.02  # 对齐容差（秒）
FPS = 50       # 播放帧率
# =======================================================

def load_trajectory(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def find_camera_topics(traj):
    return sorted([k for k in traj["data"] if k.startswith("/camera_") and k.endswith("/rgb/image_raw")])

def build_index(ts_list, target_ts):
    pos = bisect_left(ts_list, target_ts)
    candidates = []
    if pos < len(ts_list):
        candidates.append((pos, abs(ts_list[pos] - target_ts)))
    if pos > 0:
        candidates.append((pos - 1, abs(ts_list[pos - 1] - target_ts)))
    if not candidates:
        return None, float("inf")
    return min(candidates, key=lambda x: x[1])

def align_frames(traj, camera_topics, tol=0.02):
    ref_topic = camera_topics[0]
    ref_ts = traj["timestamps"][ref_topic]
    others_ts = {topic: traj["timestamps"][topic] for topic in camera_topics[1:]}

    aligned = []
    for i, t0 in enumerate(ref_ts):
        indices = [i]
        ok = True
        for topic in camera_topics[1:]:
            idx, err = build_index(others_ts[topic], t0)
            if idx is None or err > tol:
                ok = False
                break
            indices.append(idx)
        if ok:
            aligned.append((indices, t0))
    return aligned

def play_video(traj, camera_topics, aligned_pairs, fps=10):
    wait_time = int(1000 / fps)
    for indices, ts_ref in aligned_pairs:
        imgs = []
        for i, topic in enumerate(camera_topics):
            frame = traj["data"][topic][indices[i]]
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            resized = cv2.resize(frame, (480, 360))
            imgs.append(resized)

        concat_img = cv2.hconcat(imgs)
        cv2.putText(concat_img, f"{ts_ref:.3f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Synchronized Camera Playback", concat_img)
        key = cv2.waitKey(wait_time)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    traj = load_trajectory(PKL_PATH)
    camera_topics = find_camera_topics(traj)
    if not camera_topics:
        print("❌ No camera image topics found.")
        return

    aligned = align_frames(traj, camera_topics, tol=TIMESTAMP_TOL)
    if not aligned:
        print("❌ No aligned frames found.")
        return

    print(f"✅ Playing {len(aligned)} synchronized frames at {FPS} FPS...")
    play_video(traj, camera_topics, aligned, fps=FPS)

if __name__ == "__main__":
    import pickle, pprint

    with open(PKL_PATH, "rb") as f:
        bag = pickle.load(f)

    import pprint
    pprint.pprint(bag, depth=2)  # 控制嵌套层级


    print("Available top‑level keys:", bag.keys())  # 应该有 meta/data/timestamps
    print("Sample data keys:")
    pprint.pprint(list(bag["data"].keys())[:10])

    main()
