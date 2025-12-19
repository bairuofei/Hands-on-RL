#!/usr/bin/env python3
"""
硬编码路径打开 MuJoCo XML 模型并交互显示
"""

import os
import mujoco
import mujoco.viewer

def main():
    # 1. 直接写死你的模型文件路径
    xml_path = os.path.expanduser("./r1_pro.xml")  # ← 修改为你的实际文件路径

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 打开交互式窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 模拟一步
            mujoco.mj_step(model, data)

            # 将状态刷新到窗口
            viewer.sync()

if __name__ == "__main__":
    main()
