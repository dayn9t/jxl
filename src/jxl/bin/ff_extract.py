from pathlib import Path
import subprocess
import logging
from typing import Union, List


def extract_frames_from_mkv_using_ffmpeg(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    frame_rate: Union[int, str] = 1,
) -> List[Path]:
    """查找目录内所有MKV文件，使用ffmpeg解码为图片并保存

    Args:
        src_dir: 源目录，包含MKV文件
        dst_dir: 目标目录，用于保存提取的图片
        frame_rate: 提取帧率，可以是数字(如1表示每秒1帧)或ffmpeg格式字符串(如"1/10"表示每10秒1帧)

    Returns:
        List[Path]: 处理的MKV文件路径列表
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # 确保目标目录存在
    dst_path.mkdir(parents=True, exist_ok=True)

    # 查找所有MKV文件
    mkv_files = list(src_path.glob("*.mkv"))
    logging.info(f"找到 {len(mkv_files)} 个MKV文件")

    processed_files: List[Path] = []

    # 处理每个MKV文件
    for mkv_file in mkv_files:
        file_name = mkv_file.stem  # 获取不带扩展名的文件名
        output_dir = dst_path / file_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"处理文件: {mkv_file}")
        processed_files.append(mkv_file)

        # 构建ffmpeg命令
        output_pattern = str(output_dir / f"%04d.jpg")

        cmd = [
            "ffmpeg",
            "-i",
            str(mkv_file),
            "-vf",
            f"fps={frame_rate}",
            "-q:v",
            "2",  # 图片质量设置，2表示高质量
            output_pattern,
        ]

        # 执行ffmpeg命令
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logging.info(f"文件 {mkv_file.name} 处理完成")
        except subprocess.CalledProcessError as e:
            logging.error(f"处理文件 {mkv_file.name} 时出错: {e.stderr}")

    return processed_files


def main():
    logging.basicConfig(level=logging.INFO)

    src_dir = "/mnt/temp/2025_03_24"  # 替换为源MKV文件目录
    dst_dir = "/var/howell/s4/ias/track/2c"  # 替换为输出图片目录
    fps = 1.25  # 每秒1帧

    # 调用函数提取帧
    processed_files = extract_frames_from_mkv_using_ffmpeg(src_dir, dst_dir, fps)

    logging.info(f"总共处理了 {len(processed_files)} 个MKV文件")


if __name__ == "__main__":
    main()
