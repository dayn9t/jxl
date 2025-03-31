from jcx.sys.fs import *


def main():
    src_dir = "/var/howell/s4/ias/snapshot/d1/n1/13/0"  # 示例源目录
    dirs = dirs_in(src_dir)

    n = 0
    for folder in dirs:
        # 获取当前目录下的所有文件
        files = files_in(folder, ".jpg")
        n += len(files)
        print("", folder.name, len(files))
    print("total:", n)


if __name__ == "__main__":
    main()
