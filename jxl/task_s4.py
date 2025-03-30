from time import sleep
from typing import List, Optional

from jcx.db.jdb.table import *
from jcx.text.txt_json import load_json, to_json
from jvi.geo.point2d import Point


class D2dParams(Record):
    """D2D参数配置"""

    fps: Optional[float]
    roi: List[Point]


class TaskInfo(Record):
    name: str
    data_urls: List[str]
    type: int
    group_id: int
    created_at: str


class StatusInfo(Record):
    status: int
    progress: int
    start_time: str
    end_time: Optional[str]
    collection_date: str


class TaskDb:

    def __init__(self, db_dir: Path):
        self.task_tab = Table(TaskInfo)
        self.status_tab = Table(StatusInfo)
        self.task_tab.load(db_dir / "task")
        self.status_tab.load(db_dir / "status")

    def show(self):
        for task in self.task_tab.records():
            print(to_json(task))
        for status in self.status_tab.records():
            print(to_json(status))

    def find_task(self) -> Option[TaskInfo]:
        """找到可执行任务"""
        for status in self.status_tab.records():
            if status.progress < 10:
                return self.task_tab.get(status.id)
        return Null

    def task_done(self, task_id: int):
        """终结任务"""
        status: StatusInfo = self.status_tab.get(task_id).unwrap()
        status.progress = 10
        self.status_tab.update(status)


def main():
    db_dir = Path("/opt/howell/s4/current/ias/domain/d1/n1/db")
    dst_dir = Path("/var/howell/s4/ias/track")

    db = TaskDb(db_dir)
    # db.show()

    while True:
        task = db.find_task()
        match task:
            case Some(task):
                print("find task:", to_json(task))
                db.task_done(task.id)
            case None:
                print("no task")
        sleep(5)


if __name__ == "__main__":
    main()
