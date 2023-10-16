import cv2
import numpy as np
from numpy import ndarray


def p2jpg(digman, p, x1, y1, x2, y2, points_68, full_frame, color_comple):
    """单个推理图片到整帧转换"""
    p = p.astype(np.uint8)
    # 转分辨率到原人脸检测长方形
    height, width = y2 - y1, x2 - x1
    if p.shape[:2] != (height, width):
        p = cv2.resize(p, (width, height))
    try:
        if isinstance(points_68, ndarray) and points_68.shape[0] >= 17:
            # 面部轮廓
            face_points = points_68[:17]
            if points_68.shape[0] >= 25:
                face_points = np.append(face_points, [points_68[24], points_68[19]], axis=0)
            face_points = np.stack(face_points).astype(np.int32)
            # 1. 创建一个长方形遮罩
            mask = np.zeros(p.shape[:2], dtype=np.uint8)
            # 2. 使用fillPoly绘制人脸遮罩
            cv2.fillPoly(mask, [face_points], (255, 255, 255))
            # 反向遮罩
            reverse_mask = cv2.bitwise_not(mask)
            # 3. 使用遮罩提取人脸
            face_image = cv2.bitwise_and(p, p, mask=mask)
            # 原帧对应部分
            org_face_rect = full_frame[y1:y2, x1:x2]
            # 提取人脸周围
            face_surrounding = cv2.bitwise_and(org_face_rect, org_face_rect, mask=reverse_mask)
            # 推理出的人脸贴回原帧
            inferenced_face_rect = cv2.add(face_image, face_surrounding)
        else:
            # 将推理的人脸覆盖原帧
            inferenced_face_rect = p
    except:
        # 将推理的人脸覆盖原帧
        full_frame[y1:y2, x1:x2] = p
    else:
        # 将推理的人脸覆盖原帧
        full_frame[y1:y2, x1:x2] = inferenced_face_rect

    if digman in color_comple:
        full_frame = np.clip(
            full_frame + np.array(color_comple[digman]), 0, 255).astype(np.uint8)

    # 将帧转换为JPEG格式。
    ret, buffer = cv2.imencode(".jpg", full_frame)
    if ret:
        return (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
