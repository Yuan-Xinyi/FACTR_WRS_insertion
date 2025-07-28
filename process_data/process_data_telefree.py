# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import yaml
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

from utils_data_process import sync_data_slowest, process_rgb_image, gaussian_norm, generate_robobuf

@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig):

    input_path = cfg.input_path
    output_path = cfg.output_path
    
    rgb_obs_topics = list(cfg.cameras_topics)
    state_obs_topics = list(cfg.obs_topics)
    
    assert len(state_obs_topics) > 0, "Require low-dim observation topics"
    assert len(rgb_obs_topics) > 0, "Require visual observation topics"

    data_folder = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_topics = state_obs_topics + rgb_obs_topics
    all_episodes = sorted([f for f in data_folder.iterdir() if f.name.startswith('trajectory_') and f.name.endswith('.pkl')])
    
    trajectories = []
    all_states = []

    pbar = tqdm(all_episodes)
    for episode_pkl in pbar:
        with open(episode_pkl, 'rb') as f:
            full_data = pickle.load(f)

        # ✅ 跳过同步步骤，直接用你预同步的 data 字典
        traj_data = full_data['data']

        traj = {}
        num_steps = len(traj_data[state_obs_topics[0]])
        traj['num_steps'] = num_steps

        # ✅ 拼接状态向量
        traj['states'] = np.concatenate([np.array(traj_data[topic]) for topic in state_obs_topics], axis=-1)
        all_states.append(traj['states'])

        # ✅ 图像处理（resize 和压缩）
        for cam_ind, topic in enumerate(rgb_obs_topics):
            raw_images = traj_data[topic]  # 每步是 (480, 640, 3)
            processed_images = [process_rgb_image(img) for img in raw_images]
            traj[f'enc_cam_{cam_ind}'] = processed_images

            # import cv2
            # for i in range(min(3, len(processed_images))):
            #     img_bytes = processed_images[i]
            #     img_array = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            #     cv2.imshow(f"Camera {cam_ind} - Frame {i}", img_array)
            #     cv2.waitKey(0)  # 等待按键
            # cv2.destroyAllWindows()

        trajectories.append(traj)

    # ✅ 状态归一化
    state_norm_stats = gaussian_norm(all_states)
    norm_stats = dict(state=state_norm_stats)

    # ✅ 生成无动作的 robobuf
    buffer_name = "buf"
    buffer = generate_robobuf(trajectories, include_action=False)
    with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
        pickle.dump(buffer.to_traj_list(), f)

    # ✅ 保存配置（无 action）
    obs_config = {
        'state_topics': state_obs_topics,
        'camera_topics': rgb_obs_topics,
    }
    rollout_config = {
        'obs_config': obs_config,
        'action_config': None,
        'norm_stats': norm_stats
    }
    with open(output_dir / "rollout_config.yaml", "w") as f:
        yaml.dump(rollout_config, f, sort_keys=False)

        
if __name__ == "__main__":
    main()