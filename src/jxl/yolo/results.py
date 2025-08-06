from typing import List

from ultralytics.engine.results import Results

from jxl.det.d2d import D2dResult
from jxl.det.yolo.adapter import boxes_to_d2d


def results_to_d2d_result(results: Results) -> D2dResult:
    """检测结果转换为D2dResult"""
    assert isinstance(results, Results)
    objects = boxes_to_d2d(results.boxes)
    return D2dResult(objects=objects)


def results_list_to_d2d_result(results_list: List[Results]) -> D2dResult:
    """检测结果列表转换为D2dResult, 单帧图像的检测结果"""
    assert isinstance(results_list, list)
    assert len(results_list) == 1
    return results_to_d2d_result(results_list[0])


def results_list_to_d2d_results(results_list: List[Results]) -> List[D2dResult]:
    """检测结果列表转换为D2dResult, 多帧图像的检测结果"""
    assert isinstance(results_list, list)
    return [results_to_d2d_result(rs) for rs in results_list]
